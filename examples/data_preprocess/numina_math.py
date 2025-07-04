"""
Preprocess the Numia dataset to parquet format
"""

import os
import datasets
from transformers import AutoTokenizer

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/numina_math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_end', type=int, default=10000000)
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-Math-1.5B')

    args = parser.parse_args()

    data_source = 'ScaleML-RLHF/numina_math'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    train_dataset = dataset['train']
    args.train_end = min(args.train_end, len(train_dataset))
    if args.train_end > 0:
        train_dataset = train_dataset.select(range(args.train_start, args.train_end))

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('problem')

            question = question_raw + ' ' + instruction_following

            answer_raw = example['answer']
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn
    
    def able_to_extract(example):
        if len(tokenizer.encode(example['problem'])) > 700:
            return False
        return True

    train_dataset = train_dataset.filter(able_to_extract)
    
    print(f"Train dataset size: {len(train_dataset)}")

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    print(train_dataset[0])
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
                                              
