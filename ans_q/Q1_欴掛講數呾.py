import math  # 导入数学模块以使用数学函数

# 定义计算样本量的函数
def calculate_sample_size(p, z, e):
    # 计算样本量的公式：n = (z^2 * p * (1 - p)) / (e^2)
    # p: 标称次品率
    # z: 置信水平对应的z值
    # e: 允许误差
    return math.ceil((z**2 * p * (1 - p)) / (e**2))  # 向上取整以确保样本量为整数

# 参数设置
p = 0.1  # 标称次品率 10%
e = 0.05  # 允许误差 5%

# 95% 置信水平
z_95 = 1.96  # 95%置信水平对应的z值
sample_size_95 = calculate_sample_size(p, z_95, e)  # 计算95%置信水平下的样本量

# 90% 置信水平
z_90 = 1.645  # 90%置信水平对应的z值
sample_size_90 = calculate_sample_size(p, z_90, e)  # 计算90%置信水平下的样本量

# 输出结果
print(f"在95%置信水平下，最少需要抽样检测 {sample_size_95} 个零配件")  # 打印95%置信水平下的样本量
print(f"在90%置信水平下，最少需要抽样检测 {sample_size_90} 个零配件")  # 打印90%置信水平下的样本量