#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
import re
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
import json
import os
import torch.multiprocessing as mp

@dataclass
class ScriptArguments:
    """
    数据生成脚本的参数
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "SFT模型的路径"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="ScaleML-RLHF/numina_math",
        metadata={"help": "numina_math数据集路径"},
    )
    output_dir: Optional[str] = field(
        default="./training_data/",
        metadata={"help": "输出目录"},
    )
    my_world_size: Optional[int] = field(
        default=1,
        metadata={"help": "进程总数"},
    )
    K: Optional[int] = field(
        default=16,
        metadata={"help": "每个问题生成的回答数量"},
    )
    max_input_length: Optional[int] = field(
        default=2048,
        metadata={"help": "输入token的最大长度"},
    )
    max_new_tokens: Optional[int] = field(
        default=4096,
        metadata={"help": "新生成token的最大长度"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "随机种子"},
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "采样温度"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "是否使用beam search"},
    )


def worker_process(rank, world_size, script_args, prompts_chunk, questions_chunk, answers_chunk):
    """每个GPU进程的工作函数"""
    # 设置CUDA设备
    torch.cuda.set_device(rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    
    model_path = script_args.model_name_or_path
    print(f"进程 {rank}: 加载模型到GPU {rank}")
    
    # 设置随机种子
    torch.manual_seed(script_args.seed + rank)
    np.random.seed(script_args.seed + rank)
    
    # 加载模型
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        max_model_len=script_args.max_input_length,
        load_format="auto",
        seed=42+rank,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        top_p=1.0,
        max_tokens=script_args.max_new_tokens,
        n=script_args.K,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    
    print(f"进程 {rank}: 开始生成回答，处理 {len(prompts_chunk)} 个样本")
    outputs = llm.generate(prompts_chunk, sampling_params=sampling_params, use_tqdm=True)
    
    # 处理结果，保存所有生成的数据
    training_data = []
    
    for i, output in enumerate(outputs):
        responses = [out.text for out in output.outputs] 
        # 创建训练样本
        training_sample = {
            "question": questions_chunk[i],
            "answer": answers_chunk[i],
            "responses": responses  # 保存生成的回答
        }
        training_data.append(training_sample)
    
    # 保存当前进程的训练数据
    os.makedirs(script_args.output_dir, exist_ok=True)
    output_file = os.path.join(script_args.output_dir, f"training_data_rank_{rank}.jsonl")
    with open(output_file, "w", encoding="utf8") as f:
        for data in training_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def merge_results(script_args, world_size):
    """合并所有进程的结果"""
    print("开始合并所有进程的训练数据...")
    
    all_training_data = []
    
    # 读取所有进程的结果
    for rank in range(world_size):
        result_file = os.path.join(script_args.output_dir, f"training_data_rank_{rank}.jsonl")
        
        # 读取训练数据
        with open(result_file, "r", encoding="utf8") as f:
            for line in f:
                all_training_data.append(json.loads(line))
        
        # 删除临时文件
        os.remove(result_file)
    
    # 保存合并后的训练数据
    output_file = os.path.join(script_args.output_dir, f"training_data.jsonl")
    with open(output_file, "w", encoding="utf8") as f:
        for data in all_training_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    
    print(f"训练数据生成完成！")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # 检测可用GPU数量
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个GPU")
    
    if world_size == 0:
        raise RuntimeError("没有检测到可用的GPU")
    
    model_path = script_args.model_name_or_path
    print("模型路径:", model_path)
    
    # 加载tokenizer用于数据预处理
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 加载numina_math数据集的训练集
    print(f"加载numina_math数据集: {script_args.dataset_name_or_path}")
    dataset = load_dataset(script_args.dataset_name_or_path, trust_remote_code=True)
    ds = dataset['train']
    
    # 添加说明文本
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    
    # 处理数据集
    def process_numina_data(example):
        question = example["problem"] + " " + instruction_following
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        
        return {
            "prompt": prompt,
            "question": example["problem"],
            "answer": example["answer"],
        }
    
    ds = ds.map(process_numina_data)
    
    # 准备数据
    prompts = ds["prompt"]
    questions = ds["question"]
    answers = ds["answer"]
    
    # 将数据分割给各个进程
    chunk_size = len(prompts) // world_size
    processes = []
    
    for rank in range(world_size):
        start_idx = rank * chunk_size
        if rank == world_size - 1:  # 最后一个进程处理剩余的所有数据
            end_idx = len(prompts)
        else:
            end_idx = (rank + 1) * chunk_size
        
        prompts_chunk = prompts[start_idx:end_idx]
        questions_chunk = questions[start_idx:end_idx]
        answers_chunk = answers[start_idx:end_idx]
        
        print(f"进程 {rank}: 处理索引 {start_idx} 到 {end_idx-1} ({len(prompts_chunk)} 个样本)")
        
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, script_args, prompts_chunk, questions_chunk, answers_chunk)
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 合并结果
    merge_results(script_args, world_size)
