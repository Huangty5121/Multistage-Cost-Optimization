from scipy.stats import binom #导入二项分布相关的函数
#初步筛选阶段
# 参数定义
n_50 = 98  # 第一阶段抽样数量
p_10 = 0.10  # 标称次品率
alpha_95 = 0.05  # 95%置信水平
alpha_90 = 0.10  # 90%置信水平

# 计算95%置信水平下拒收的临界次品数量
k_reject_95 = binom.ppf(1 - alpha_95, n_50, p_10)#使用ppf函数计算拒收的临界值

# 计算90%置信水平下接收的临界次品数量
k_accept_90 = binom.ppf(alpha_90, n_50, p_10)#使用ppf函数计算接收的临界值

print(k_reject_95, k_accept_90)  # 输出拒收和接收的次品数量

# 详细检测阶段：样本量 139
n_139 = 139  # 第二阶段抽样数量

# 计算在139个样本中，95%置信水平下拒收的临界次品数量
k_reject_139_95 = binom.ppf(1 - alpha_95, n_139, p_10)  # 计算拒收的临界值

# 计算在139个样本中，90%置信水平下接收的临界次品数量
k_accept_139_90 = binom.ppf(alpha_90, n_139, p_10)  # 计算接收的临界值

print(k_reject_139_95, k_accept_139_90)  # 输出拒收和接收的次品数量

# 模型的推广与拓展：计算不同样本量下的临界次品数量
sample_sizes = range(50, 126, 15)  # 样本量从50到125，步长为15

for n in sample_sizes:
    # 计算在不同样本量下，95%置信水平下拒收的临界次品数量
    k_reject_95 = binom.ppf(1 - alpha_95, n, p_10)  # 计算拒收的临界值
    # 计算在不同样本量下，90%置信水平下接收的临界次品数量
    k_accept_90 = binom.ppf(alpha_90, n, p_10)  # 计算接收的临界值
    # 输出当前样本量下的拒收和接收临界次品数量
    print(f"样本量 {n}: 拒收临界次品数量 = {k_reject_95}, 接收临界次品数量 = {k_accept_90}")