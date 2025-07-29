import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置全局字体为Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14  # 增大基础字体大小

def plot_token_prob_distribution_from_multiple_files(json_paths, name_map, img_path, dpo_data=None):
    print("\n📊 Comparing token probability distributions from:")
    if dpo_data:
        print(f"  - DPO (direct data)")
    for path in json_paths:
        print(f"  - {path}  ({name_map.get(path, 'Unnamed')})")

    # 自定义 bin 划分：低概率区域更细
    bins = [0.0, 0.0001, 0.001, 0.01, 0.1,
            0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]
    # 增加横坐标之间的间距
    x = np.arange(len(labels)) * 1.25  # 将间距从1增加到1.5

    # 图形设置：更宽 + 更清晰
    plt.figure(figsize=(16, 10))  # 增大图形尺寸以适应更多模型
    bar_width = 0.15  # 减小条形宽度以适应更多模型
    
    # 计算模型数量（包括DPO）
    num_models = len(json_paths) + (1 if dpo_data else 0)
    offsets = np.linspace(-bar_width * (num_models-1)/2, bar_width * (num_models-1)/2, num_models)

    # colors = ['#E4F6E7', '#D6E9F5', '#FFF2CC', '#F3DEE0', '#F3F8F9', '#7B8BC9']  # 浅粉、浅绿、浅蓝、浅紫、浅黄、浅橙

    colors = [
        '#B7E4C7',  # Base - 浅蓝
        '#D6E9F5',  # SFT - 浅绿
        '#FFE9B1',  # DFT - 青绿（Base与SFT中间调）
        '#FFF2B2',  # PPO - 浅金黄
        '#FAD7A0',  # GRPO - 浅橙
        '#7B8BC9',  # DPO - 浅粉橘
    ]



    current_offset_index = 0
    
    # 绘制前两个模型的数据
    for i in range(min(2, len(json_paths))):
        json_path = json_paths[i]
        with open(json_path, "r") as f:
            results = json.load(f)

        probs = []
        for entry in results:
            probs.extend(entry.get("probs", []))

        hist, _ = np.histogram(probs, bins=bins)
        total = sum(hist)
        percentages = hist / total * 100 if total > 0 else np.zeros_like(hist)

        model_name = name_map.get(json_path, json_path)
        plt.bar(x + offsets[current_offset_index], percentages,
                width=bar_width,
                label=model_name,
                alpha=0.8,
                color=colors[current_offset_index],
                edgecolor='gray',
                linewidth=0.8)

        # 输出统计信息
        print(f"\n📌 Model: {model_name}")
        print(f"Total tokens: {total}")
        for label, count in zip(labels, hist):
            percent = count / total * 100 if total > 0 else 0.0
            print(f"  {label:>12}: {count:6d} tokens ({percent:6.2f}%)")
        
        current_offset_index += 1

    # 绘制DPO数据（第三个位置）
    if dpo_data:
        # DPO数据：百分比值
        dpo_percentages = [1.262, 0.832, 1.837, 5.024, 3.096, 2.633, 2.348, 2.436, 2.517, 2.827, 3.472, 5.052, 66.664]
        
        plt.bar(x + offsets[current_offset_index], dpo_percentages,
                width=bar_width,
                label="DPO",
                alpha=0.8,
                color=colors[current_offset_index],
                edgecolor='gray',
                linewidth=0.8)

        # 输出DPO统计信息
        print(f"\n📌 Model: DPO")
        print(f"Total tokens: 847,803")
        dpo_counts = [10701, 7048, 15575, 42584, 26246, 22322, 19898, 20644, 21334, 23959, 29427, 42825, 565058]
        for label, count, percent in zip(labels, dpo_counts, dpo_percentages):
            print(f"  {label:>12}: {count:6d} tokens ({percent:6.2f}%)")
        
        current_offset_index += 1

    # 绘制剩余模型的数据
    for i in range(2, len(json_paths)):
        json_path = json_paths[i]
        with open(json_path, "r") as f:
            results = json.load(f)

        probs = []
        for entry in results:
            probs.extend(entry.get("probs", []))

        hist, _ = np.histogram(probs, bins=bins)
        total = sum(hist)
        percentages = hist / total * 100 if total > 0 else np.zeros_like(hist)

        model_name = name_map.get(json_path, json_path)
        plt.bar(x + offsets[current_offset_index], percentages,
                width=bar_width,
                label=model_name,
                alpha=0.8,
                color=colors[current_offset_index],
                edgecolor='gray',
                linewidth=0.8)

        # 输出统计信息
        print(f"\n📌 Model: {model_name}")
        print(f"Total tokens: {total}")
        for label, count in zip(labels, hist):
            percent = count / total * 100 if total > 0 else 0.0
            print(f"  {label:>12}: {count:6d} tokens ({percent:6.2f}%)")
        
        current_offset_index += 1

    plt.xticks(x, labels, rotation=45, ha='right', fontsize=16)  # 增大x轴标签字体
    plt.title("Token Probability Distribution", fontsize=20, fontweight='bold')  # 增大标题字体并加粗
    plt.xlabel("Prediction Probability", fontsize=18)  # 增大x轴标签字体
    plt.ylabel("Percentage  of  Tokens (%)", fontsize=18)  # 增大y轴标签字体
    
    # 设置对数坐标轴
    plt.yscale('log')
    
    # 设置y轴刻度标签字体
    # plt.yticks(fontsize=16)
     # 设置y轴刻度：0, 0.5, 1, 5, 10, 50, 100
    y_ticks = [0.1, 0.3, 1, 3, 10, 30, 100]
    plt.yticks(y_ticks, [str(tick) for tick in y_ticks], fontsize=16)
   
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.legend(title="Method", fontsize=16, title_fontsize=18)  # 增大图例字体
    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.savefig("test.pdf", dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved figure to {img_path}")

# 用例
json_files = [
    "answer_probs_vllm_qwen25_math_base.json",
    "answer_probs_vllm_sft_step_390.json",
    "answer_probs_vllm_qwen25_math_base_grpo.json",
    "answer_probs_vllm_qwen25_math_base_ppo.json",
    "answer_probs_vllm_scale_0_step_390.json",
]

name_map = {
    "answer_probs_vllm_qwen25_math_base.json": "Base",
    "answer_probs_vllm_sft_step_390.json": "SFT",
    "answer_probs_vllm_qwen25_math_base_grpo.json": "GRPO",
    "answer_probs_vllm_qwen25_math_base_ppo.json": "PPO",
    "answer_probs_vllm_scale_0_step_390.json": "DFT",
}

# DPO数据
dpo_data = {
    "percentages": [1.262, 0.832, 1.837, 5.024, 3.096, 2.633, 2.348, 2.436, 2.517, 2.827, 3.472, 5.052, 66.664],
    "counts": [10701, 7048, 15575, 42584, 26246, 22322, 19898, 20644, 21334, 23959, 29427, 42825, 565058]
}

output_figure = "combined_token_distribution.jpg"
plot_token_prob_distribution_from_multiple_files(json_files, name_map, output_figure, dpo_data)
