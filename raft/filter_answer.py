#!/usr/bin/env python
import json
import os
from tqdm import tqdm
from math_verify import parse, verify
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

def extract_boxed_answer(text):
    """从文本中提取\\boxed{}内的答案"""
    try:
        boxed_string = last_boxed_only_string(text)
        if boxed_string is None:
            return None
        else:
            return remove_boxed(boxed_string)
    except Exception as e:
        return None

def verify_math_answer(generated_answer, ground_truth_answer):
    """验证生成的答案是否与标准答案匹配"""
    # 从生成的回答中提取boxed答案
    extracted_answer = extract_boxed_answer(generated_answer)
    if extracted_answer is None:
        return False
    
    # 从标准答案中提取boxed答案
    gt_extracted = extract_boxed_answer(ground_truth_answer)
    if gt_extracted is None:
        # 如果标准答案没有boxed格式，直接使用原答案
        gt_extracted = ground_truth_answer
    
    gold = parse(extracted_answer)
    answer = parse(gt_extracted)

    if verify(gold, answer):
        return True
    else:
        return False

def process_sample(sample_data):
    """处理单个样本的函数，用于多线程"""
    question, ground_truth, responses = sample_data
    correct_samples = []
    
    for response in responses:
        if verify_math_answer(response, ground_truth):
            # 每个正确回答作为一个单独的训练样本
            filtered_sample = {
                "question": question,
                "answer": ground_truth,
                "response": response
            }
            correct_samples.append(filtered_sample)
    
    return correct_samples, len(responses), len(correct_samples)

def filter_correct_answers(input_file, output_file, num_threads=8):
    """筛选出正确答案的训练数据，使用多线程并行验证"""
    print(f"开始筛选正确答案，输入文件: {input_file}")
    print(f"使用 {num_threads} 个线程进行并行验证")
    
    # 读取所有数据
    all_samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            all_samples.append((data['question'], data['answer'], data['responses']))
    
    print(f"总共加载 {len(all_samples)} 个样本")
    
    # 使用线程池进行并行处理
    filtered_data = []
    total_samples = len(all_samples)
    total_responses = 0
    correct_responses = 0
    
    # 创建线程锁用于安全地更新共享变量
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_sample = {executor.submit(process_sample, sample): sample for sample in all_samples}
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(future_to_sample), total=len(all_samples), desc="验证答案"):
            try:
                correct_samples, num_responses, num_correct = future.result()
                
                # 线程安全地更新统计信息
                with lock:
                    filtered_data.extend(correct_samples)
                    total_responses += num_responses
                    correct_responses += num_correct
                    
            except Exception as exc:
                print(f'样本处理出现异常: {exc}')
    
    # 保存筛选后的数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in filtered_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # 打印统计信息
    print(f"筛选完成！")
    print(f"总样本数: {total_samples}")
    print(f"总生成回答数: {total_responses}")
    print(f"正确回答数: {correct_responses}")
    print(f"正确率: {correct_responses/total_responses*100:.2f}%")
    print(f"筛选后的训练样本数: {len(filtered_data)}")
    print(f"筛选后数据保存到: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./training_data_k4/training_data.jsonl")
    parser.add_argument("--output_file", type=str, default="./training_data_k4/filtered_training_data.jsonl")
    parser.add_argument("--num_threads", type=int, default=64, help="并行验证的线程数")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    num_threads = args.num_threads
    filter_correct_answers(input_file, output_file, num_threads)