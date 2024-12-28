import math
from scipy.stats import norm, binom

# 计算样本量的函数
def calculate_sample_size(p: float, z: float, e: float) -> int:
    """计算样本量：n = (z^2 * p * (1 - p)) / (e^2)"""
    n = (z**2 * p * (1 - p)) / (e**2)
    return math.ceil(n)  # 向上取整以确保样本量为整数

# 计算 Z 值的函数（基于置信度）
def calculate_z_value(confidence_level: float) -> float:
    """根据置信度计算 Z 值。"""
    z_value = norm.ppf(1 - (1 - confidence_level) / 2)
    print(f"Calculated Z value for {int(confidence_level * 100)}% confidence: {z_value}")
    return z_value

# 计算拒收和接收临界次品数量
def calculate_critical_values(n: int, p: float, confidence_level: float) -> tuple:
    """计算拒收和接收的临界次品数量。"""
    alpha = 1 - confidence_level  # 显著性水平计算
    k_reject = binom.ppf(1 - alpha, n, p)  # 拒收的临界值
    k_accept = binom.ppf(alpha, n, p)  # 接收的临界值
    return k_reject, k_accept

# 抽样检测方案的实现
def sampling_plan(p: float, e: float, confidence_levels: list):
    """
    设计抽样检测方案，计算每个置信度下的样本量、拒收和接收的临界次品数量。
    
    参数：
    p (float): 标称次品率
    e (float): 允许误差
    confidence_levels (list): 置信度列表
    """
    for confidence_level in confidence_levels:
        z_value = calculate_z_value(confidence_level)  # 根据置信度计算Z值
        sample_size = calculate_sample_size(p, z_value, e)  # 计算样本量
        print(f"在{int(confidence_level * 100)}%置信水平下，最少需要抽样检测 {sample_size} 个零配件")
        
        # 计算拒收和接收的临界次品数量
        k_reject, k_accept = calculate_critical_values(sample_size, p, confidence_level)
        
        print(f"在{int(confidence_level * 100)}%置信水平下：")
        print(f"拒收临界次品数量：{k_reject}")
        print(f"接收临界次品数量：{k_accept}")
        print("-" * 50)

# 主函数，程序的入口
def main():
    """
    主函数，调用抽样检测方案的设计函数，计算所需的样本量及临界次品数量。
    """
    # 用户输入参数
    p = 0.1  # 标称次品率 10%
    e = 0.05  # 允许误差 5%
    confidence_levels = [0.95, 0.90]  # 95% 和 90% 置信度
    
    # 调用函数进行抽样检测方案设计
    sampling_plan(p, e, confidence_levels)

# 调用主函数
if __name__ == "__main__":
    main()
