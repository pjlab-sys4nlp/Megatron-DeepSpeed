# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain Llama"""
import torch
import math
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.data.dummy_dataset import RandomDataset
from megatron.model import LlamaModel, LlamaModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
import os
import subprocess
from megatron.model.module import MegatronModule, float16_to_fp32, fp32_to_float16


import torch
import os

from torch import nn
from torch.distributed import GroupMember
import torch.nn as nn


import torch.nn.functional as F

class ToyModel2(MegatronModule):
    def __init__(self) -> None:
        super(ToyModel2, self).__init__()
        self.step_count = 0
        self.dense1 = torch.nn.Parameter(data=torch.tensor([-i for i in range(64)], dtype=torch.bfloat16, requires_grad=True))

    def forward(self, x, labels=None):    
        x = x.to(f"cuda:{int(os.environ['SLURM_PROCID']) % 8}")
        if os.environ['SLURM_PROCID'] == '0':
            print(f"Rank: {os.environ['SLURM_PROCID']}:\
do ToyModel2-forward!,self.step_count: {self.step_count}, \
self.dense:{self.dense1}", flush=True)

        self.step_count +=1 
        y = self.dense1.mul(x)
        # x =  self.dense2.mul(y)
        return y

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

class ToyModel(MegatronModule):
    def __init__(self) -> None:
        # 64 / 16 = 4
        # 64 / 8 = 8
        super(ToyModel, self).__init__()
        self.step_count = 0
        # self.dense = torch.nn.Linear(64, 1, bias=False, dtype=torch.bfloat16)
        # 我们让 reduce_bucket_size 正好等于 64，这样在做完 ToyModel2 的 bwd 后就可以直接进行 bucket 的 allreduce
        self.dense1 = torch.nn.Parameter(data=torch.tensor([i for i in range(64)], dtype=torch.bfloat16, requires_grad=True))
        self.sub_module = ToyModel2()
        self.dense2 = torch.nn.Parameter(data=torch.tensor([-i for i in range(64)], dtype=torch.bfloat16, requires_grad=True))

    def forward(self, x, labels=None):    
        x = x.to(f"cuda:{int(os.environ['SLURM_PROCID']) % 8}")
        if os.environ['SLURM_PROCID'] == '0':
            print(f"Rank: {os.environ['SLURM_PROCID']}:\
do forward!,self.step_count: {self.step_count}, \
self.dense:{self.dense1}", flush=True)

        self.step_count +=1 
        y = self.sub_module(x)
        y = self.dense1 + y
        x =  self.dense2.mul(y)
        return x

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor


from deepspeed.comm import init_distributed
import torch.distributed as dist

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building llama model ...')
    see_memory_usage(f"Before Building Model", force=True)


    init_distributed(dist_backend="nccl", \
                    auto_mpi_discovery=False, \
                    init_method=f"tcp://[{get_master_node()}]:12349", \
                    rank=int(os.environ['SLURM_PROCID']), \
                    world_size=16)

    local_group1 = dist.new_group([i for i in range(8)])
    local_group2 = dist.new_group([i for i in range(8, 16, 1)])


    def get_param_group():
        if int(os.environ['SLURM_PROCID']) < 8:
            return local_group1
        else:
            return local_group2
    json_path = "/mnt/petrelfs/wangguoteng.p/deepspeed_zero_infini/LLaMA-Megatron-DeepSpeed/exps/stage3.json"
    with deepspeed.zero.Init(data_parallel_group=GroupMember.WORLD,
                             remote_device=None,
                             config_dict_or_path=json_path,
                             enabled=True,
                             mpu=None,
                             enable_zero35=True):
        model = ToyModel()
    see_memory_usage(f"After Building Model", force=True)
    return model


def loss_func(x):
    my_loss = torch.tensor(0.001, requires_grad = True).to(f"cuda:{int(os.environ['SLURM_PROCID']) % 8}") 
    return my_loss, {'lm loss': my_loss.clone()}


def loss_func_new(output_tensor):
    # my_loss = torch.tensor(0.001, requires_grad = True).to(f"cuda:{int(os.environ['SLURM_PROCID']) % 8}") 
    losses = output_tensor.float()
    loss = torch.sum(losses.view(-1))
    return loss, {'lm loss': loss}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()
    # Get the batch.
    timers('batch-generator').start()
    tokens, labels = torch.ones((64,), dtype=torch.bfloat16), None
    timers('batch-generator').stop()

    output_tensor = model(tokens, labels=None)
    # Output_tensor stores the standard loss, loos_func calculates the total loss.
    return output_tensor, loss_func_new


def random_train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for llama ...')
    
    seq_length = 64
    train_ds = RandomDataset(num_samples=1000000, seq_len=seq_length)
    valid_ds = RandomDataset(num_samples=20000, seq_len=seq_length)
    test_ds = RandomDataset(num_samples=10000, seq_len=seq_length)
    print_rank_0("> finished creating llama datasets ...")

    return train_ds, valid_ds, test_ds

def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


def get_master_node():
    import subprocess

    if os.getenv("SLURM_JOB_ID") is None:
        raise RuntimeError("get_master_node can only used in Slurm launch!")
    result = subprocess.check_output('scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1', shell=True)
    result = result.decode("utf8").strip()
    return result


if __name__ == "__main__":
    # git_ds_info()
    import os

    os.environ['MASTER_ADDR'] = get_master_node()
    os.environ['MASTER_PORT'] = '12349'
    os.environ['LOCAL_RANK'] = str(int(os.environ['SLURM_PROCID']) % 8)
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    os.environ['WORLD_SIZE'] = '16'

    pretrain(random_train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer', 'micro_batch_size': 1, 'num_layers':1, 'train_iters': 2}, # 'world_size': 16, 'rank': str(int(os.environ['SLURM_PROCID']))
             data_post_process=None)
