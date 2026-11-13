# 文件路径: pretraining0/src/optim/opt_utils.py
import torch

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    使用 Newton-Schulz 5阶迭代计算矩阵的零次方 (即矩阵正交化)。
    这是 Muon 优化器的核心组件。
    """
    assert len(G.shape) == 2, "Newton-Schulz 只能处理 2D 矩阵"
    
    # 5阶 Newton-Schulz 迭代的标准系数
    a, b, c = (3.4445, -4.7750,  2.0315)
    
    # 为了加速计算，通常在这个步骤转换到 bfloat16
    X = G.bfloat16()
    
    # 归一化以确保谱半径严格小于1，保证迭代收敛
    X = X / (X.norm() + eps) 
    
    # 保证我们在较小的维度上进行矩阵乘法 (A = X @ X.T)
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if transposed:
        X = X.T
        
    return X.to(G.dtype)