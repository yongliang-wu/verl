import numpy as np
from scipy.linalg import block_diag

def split_to_powers_of_two_binary(m):
    result = []
    qubit = []
    for i in range(m.bit_length()):
        if m & (1 << i):  # 检查第 i 位是否为 1
            qubit.append(i)
            result.append(2**i)
    return result[::-1], qubit[::-1] # 反转，让大的在前面

# 假设你有一个函数可以返回 size*size 的方阵
def get_block(size):
    # 这里举例用单位阵，你可以替换成你自己的方阵
    return np.eye(size)

def build_block_diagonal_matrix(m):
    sizes,qubit = split_to_powers_of_two_binary(m)
    print(sizes, qubit)
    blocks = [get_block(size) for size in sizes]
    # 用 numpy 的 block_diag 拼接
    A = block_diag(*blocks)
    return A

# 示例
m = 956
A = build_block_diagonal_matrix(m)
# print(A)
print(A.shape)  # (12, 12)
