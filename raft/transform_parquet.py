#!/usr/bin/env python
import json
import os
import pandas as pd
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

def count_tokens_for_sample(sample_data, tokenizer):
    """为单个样本计算token数量并判断是否需要过滤"""
    question = sample_data['question']
    response = sample_data['response']
    
    # 组合问题和答案
    full_text = question + " " + instruction_following + response
    
    # 计算token数量
    tokens = tokenizer.encode(full_text)
    token_count = len(tokens)
    
    # 返回样本数据和token数量
    return sample_data, token_count

def transform_filtered_jsonl_to_parquet(input_file, output_file, model_name_or_path, num_threads=8):
    """将已筛选的jsonl格式训练数据转换为parquet格式，并使用多线程过滤掉token数量大于1024的样本"""
    print(f"开始转换已筛选的训练数据为parquet格式，输入文件: {input_file}")
    print(f"使用 {num_threads} 个线程进行并行token统计和过滤")
    
    # 加载tokenizer
    print(f"加载tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # 读取所有数据
    all_samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            all_samples.append(data)
    
    print(f"总共加载 {len(all_samples)} 个样本")
    
    filtered_data = []
    total_samples = len(all_samples)
    filtered_out_samples = 0
    
    # 创建线程锁用于安全地更新共享变量
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_sample = {executor.submit(count_tokens_for_sample, sample, tokenizer): sample for sample in all_samples}
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(future_to_sample), total=len(all_samples), desc="统计token并过滤"):
            try:
                sample_data, token_count = future.result()
                
                # 线程安全地更新统计信息
                with lock:
                    # 过滤掉token数量大于1024的样本
                    if token_count > 1024:
                        filtered_out_samples += 1
                        continue
                    
                    question = sample_data['question']
                    ground_truth = sample_data['answer']
                    response = sample_data['response']
                    
                    # 按照numina_math.py的格式构造数据
                    filtered_sample = {
                        "data_source": "raft_numina_math",
                        "prompt": [
                            {
                                "role": "user",
                                "content": question + " " + instruction_following
                            }
                        ],
                        "ability": "math",
                        "reward_model": {"style": "rule", "ground_truth": ground_truth},
                        "extra_info": {
                            "split": "train",
                            "index": len(filtered_data),
                            "question": question,
                            "answer": response,
                        },
                    }
                    filtered_data.append(filtered_sample)
                    
            except Exception as exc:
                print(f'样本处理出现异常: {exc}')
    
    # 转换为DataFrame并保存为parquet格式
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(filtered_data)
    df.to_parquet(output_file, index=False)
    
    print(f"转换完成！")
    print(f"总样本数: {total_samples}")
    print(f"过滤掉的样本数（token > 1024）: {filtered_out_samples}")
    print(f"保留的样本数: {len(filtered_data)}")
    print(f"过滤比例: {filtered_out_samples / total_samples * 100:.2f}%")
    print(f"转换后数据保存到: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./training_data_k4/filtered_training_data.jsonl")
    parser.add_argument("--output_file", type=str, default="./training_data_k4/filtered_training_data.parquet")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--num_threads", type=int, default=64, help="并行处理的线程数")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    model_name_or_path = args.model_name_or_path
    num_threads = args.num_threads
    transform_filtered_jsonl_to_parquet(input_file, output_file, model_name_or_path, num_threads)