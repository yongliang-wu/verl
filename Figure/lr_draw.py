import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置美观风格
sns.set(style="whitegrid", palette="pastel")
plt.rcParams.update({'font.size': 16, 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'})

# 定义 Qwen 模型数据
qwen_data = {
    'LR': ['1e-6', '1e-6', '1e-5', '1e-5', '1e-4', '1e-4'],
    'Method': ['SFT', 'DFT', 'SFT', 'DFT', 'SFT', 'DFT'],
    'Avg': [14.41, 16.03, 14.19, 22.64, 16.81, 30.67],
    'Model': ['Qwen'] * 6
}

# 定义 LLaMA 模型数据
llama_data = {
    'LR': ['1e-6', '1e-6', '1e-5', '1e-5', '1e-4', '1e-4'],
    'Method': ['SFT', 'DFT', 'SFT', 'DFT', 'SFT', 'DFT'],
    'Avg': [1.10, 1.39, 1.83, 4.30, 3.84, 3.50],
    'Model': ['LLaMA'] * 6
}

# 创建 DataFrame
df_qwen = pd.DataFrame(qwen_data)
df_llama = pd.DataFrame(llama_data)

# 创建子图，调整为 A4 横向比例（11.7 x 5.2 英寸），减少左右留白
fig, axes = plt.subplots(1, 2, figsize=(17, 5.2), sharey=False)
plt.subplots_adjust(left=0.08, right=0.95, wspace=0.25)

# 使用浅色调色板
light_palette = sns.color_palette("pastel")

# Qwen 子图
sns.barplot(ax=axes[0], data=df_qwen, x='LR', y='Avg', hue='Method', palette=light_palette)
axes[0].set_title('Qwen-2.5-Math-1.5B-Base', fontsize=18, color='black')
axes[0].set_ylabel('Average Accuracy', fontsize=16, color='black')
axes[0].set_xlabel('Learning Rate', fontsize=16, color='black')
axes[0].grid(axis='y', linestyle='--', alpha=0.6)
axes[0].tick_params(colors='black', labelsize=14)

# LLaMA 子图
sns.barplot(ax=axes[1], data=df_llama, x='LR', y='Avg', hue='Method', palette=light_palette)
axes[1].set_title('LLaMA-3.2-3B-Base', fontsize=18, color='black')
axes[1].set_ylabel('Average Accuracy', fontsize=16, color='black')
axes[1].set_xlabel('Learning Rate', fontsize=16, color='black')
axes[1].grid(axis='y', linestyle='--', alpha=0.6)
axes[1].tick_params(colors='black', labelsize=14)

# 保存为jpg和pdf格式
plt.savefig('lr_draw.jpg', dpi=300)
plt.savefig('lr_draw.pdf')
plt.show()
