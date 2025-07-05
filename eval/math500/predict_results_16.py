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
import torch.distributed as dist
import torch.multiprocessing as mp

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval

def extract_solution(solution_str):
    """从MATH数据集的solution中提取答案"""
    return remove_boxed(last_boxed_only_string(solution_str))

def extract_answer(solution_str):
    """从MATH数据集的solution中提取答案"""
    try:
        s = last_boxed_only_string(solution_str)
        if s is None:
            return None
        result = remove_boxed(s)
        return result
    except Exception as e:
        return None

def evaluate_math_answer(generated_answer, ground_truth):
    """评估MATH答案是否正确"""
    if generated_answer is None:
        return False
    
    # 直接进行字符串比较（MATH数据集的答案格式比较复杂）
    return generated_answer.strip() == ground_truth.strip()

@dataclass
class ScriptArguments:
    """
    MATH评估脚本的参数
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="DigitalLearningGmbH/MATH-lighteval",
        metadata={"help": "MATH数据集路径"},
    )
    dataset_split: Optional[str] = field(
        default="test",
        metadata={"help": "数据集分割（train或test）"},
    )
    output_dir: Optional[str] = field(
        default="./math_eval_results/",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=1,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=16,
        metadata={"help": "每个问题生成的回答数量"},
    )
    max_input_length: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=4096,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )

def worker_process(rank, world_size, script_args, prompts_chunk, ground_truths_chunk, questions_chunk):
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
    
    # 评估结果
    evaluation_results = []
    total_accuracy = 0.0
    
    for i, output in enumerate(outputs):
        ground_truth = ground_truths_chunk[i]
        responses = [out.text for out in output.outputs]
        
        # 对每个生成的回答进行评估
        response_evaluations = []
        correct_count = 0
        for response in responses:
            generated_answer = extract_answer(response)
            is_correct = evaluate_math_answer(generated_answer, ground_truth)
            response_evaluations.append({
                "response": response,
                "extracted_answer": generated_answer,
                "is_correct": is_correct
            })
            if is_correct:
                correct_count += 1
        
        # 计算该样本的平均准确率
        sample_accuracy = correct_count / len(responses) if len(responses) > 0 else 0.0
        total_accuracy += sample_accuracy
        
        result = {
            "question": questions_chunk[i],
            "ground_truth": ground_truth,
            "responses": response_evaluations,
            "correct_count": correct_count,
            "total_responses": len(responses),
            "sample_accuracy": sample_accuracy
        }
        evaluation_results.append(result)
    
    # 保存当前进程的结果
    os.makedirs(script_args.output_dir, exist_ok=True)
    output_file = os.path.join(script_args.output_dir, f"math_eval_results_rank_{rank}.jsonl")
    with open(output_file, "w", encoding="utf8") as f:
        for result in evaluation_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 保存当前进程的统计信息
    average_accuracy = total_accuracy / len(outputs) if len(outputs) > 0 else 0.0
    stats = {
        "rank": rank,
        "total_count": len(outputs),
        "total_accuracy": total_accuracy,
        "average_accuracy": average_accuracy
    }
    
    stats_file = os.path.join(script_args.output_dir, f"stats_rank_{rank}.json")
    with open(stats_file, "w", encoding="utf8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"进程 {rank}: 完成，平均准确率: {average_accuracy:.4f}")

def merge_results(script_args, world_size):
    """合并所有进程的结果"""
    print("开始合并所有进程的结果...")
    
    all_results = []
    total_accuracy_sum = 0.0
    total_count = 0
    
    # 读取所有进程的结果
    for rank in range(world_size):
        result_file = os.path.join(script_args.output_dir, f"math_eval_results_rank_{rank}.jsonl")
        stats_file = os.path.join(script_args.output_dir, f"stats_rank_{rank}.json")
        
        # 读取结果
        with open(result_file, "r", encoding="utf8") as f:
            for line in f:
                all_results.append(json.loads(line))
        
        # 读取统计信息
        with open(stats_file, "r", encoding="utf8") as f:
            stats = json.load(f)
            total_accuracy_sum += stats["total_accuracy"]
            total_count += stats["total_count"]
        
        # 删除临时文件
        os.remove(result_file)
        os.remove(stats_file)
    
    # 保存合并后的结果
    output_file = os.path.join(script_args.output_dir, f"math_eval_results.jsonl")
    with open(output_file, "w", encoding="utf8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 保存汇总结果
    overall_average_accuracy = total_accuracy_sum / total_count if total_count > 0 else 0.0
    summary_file = os.path.join(script_args.output_dir, f"math_summary.json")
    summary = {
        "model_path": script_args.model_name_or_path,
        "dataset_split": script_args.dataset_split,
        "total_count": total_count,
        "total_accuracy_sum": total_accuracy_sum,
        "average_accuracy": overall_average_accuracy,
        "world_size": world_size,
        "parameters": {
            "temperature": script_args.temperature,
            "max_new_tokens": script_args.max_new_tokens,
            "K": script_args.K
        }
    }
    
    with open(summary_file, "w", encoding="utf8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"合并完成！总体平均准确率: {overall_average_accuracy:.4f}")
    print(summary)

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
    
    # 加载MATH数据集
    print(f"加载MATH数据集: {script_args.dataset_name_or_path}")
    ds = load_dataset(script_args.dataset_name_or_path, trust_remote_code=True, split=script_args.dataset_split)
    
    # 添加说明文本
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    
    # 处理数据集
    def process_math_data(example):
        question = example["problem"] + " " + instruction_following
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 提取ground truth答案
        ground_truth = extract_solution(example["solution"])
        
        return {
            "prompt": prompt,
            "question": example["problem"],
            "solution": example["solution"],
            "ground_truth": ground_truth
        }
    
    ds = ds.map(process_math_data)
    
    # 准备数据
    prompts = ds["prompt"]
    ground_truths = ds["ground_truth"]
    questions = ds["question"]
    
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
        ground_truths_chunk = ground_truths[start_idx:end_idx]
        questions_chunk = questions[start_idx:end_idx]
        
        print(f"进程 {rank}: 处理索引 {start_idx} 到 {end_idx-1} ({len(prompts_chunk)} 个样本)")
        
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, script_args, prompts_chunk, ground_truths_chunk, questions_chunk)
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 合并结果
    merge_results(script_args, world_size)
