# 导入必要的库
import pandas as pd  # 数据处理库
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
from scipy.stats import norm  # 统计分布库

# 设置matplotlib绘图时可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 使得坐标轴负号显示正常

# 定义问题中的数据
data = {
    '零配件': ['零配件1', '零配件2', '零配件3', '零配件4', '零配件5', '零配件6', '零配件7', '零配件8'],
    '购买单价': [2, 8, 12, 2, 8, 12, 8, 12],
    '检测成本': [1, 1, 2, 1, 1, 2, 1, 2]
}

half_products = {
    '半成品': ['半成品1', '半成品2', '半成品3'],
    '装配成本': [8, 8, 8],
    '检测成本': [4, 4, 4],
    '拆解费用': [6, 6, 6]
}

final_product = {
    '成品': ['成品'],
    '装配成本': [8],
    '检测成本': [6],
    '拆解费用': [10],
    '市场售价': [200],
    '调换损失': [40]
}

# 将数据转换为 DataFrame 格式
df_parts = pd.DataFrame(data)  # 零配件数据
df_half_products = pd.DataFrame(half_products)  # 半成品数据
df_final_product = pd.DataFrame(final_product)  # 成品数据

# 定义循环终止条件
MAX_CYCLES = 10  # 最大循环次数
EPSILON = 0.02  # 次品率阈值为2%

# 定义次品率递减比例
ALPHA_PARTS = 0.7  # 零配件次品率递减比例
ALPHA_HALF_PRODUCTS = 0.7  # 半成品次品率递减比例
ALPHA_FINAL_PRODUCT = 0.7  # 成品次品率递减比例

# 抽样检测中的参数
Z_ALPHA_95 = 1.96  # 95% 置信度下的 Z 值
Z_ALPHA_90 = 1.645  # 90% 置信度下的 Z 值
ERROR_MARGIN = 0.02  # 允许的误差

# 基于抽样检测的次品率估计函数
def estimate_defect_rate(p_estimate, confidence_level=0.95, error_margin=ERROR_MARGIN):
    z_value = norm.ppf((1 + confidence_level) / 2)  # 计算Z值
    sample_size = (z_value ** 2 * p_estimate * (1 - p_estimate)) / (error_margin ** 2)  # 计算样本大小
    # 从样本估计次品率，样本次品率为给定的 p_estimate，但添加一些随机波动
    sample_defect_rate = np.random.binomial(int(sample_size), p_estimate) / sample_size  # 计算次品率
    return sample_defect_rate  # 返回估计的次品率

# 定义总成本计算函数，并基于抽样检测估计次品率
def calculate_total_cost_with_sampling(df_parts, df_half_products, df_final_product, confidence_level=0.95,
                                       epsilon=EPSILON, max_cycles=MAX_CYCLES):
    total_cost = 0  # 初始化总成本
    cycle = 0  # 循环计数

    # 初始化次品率，通过抽样估计次品率
    part_defects = df_parts.apply(lambda x: estimate_defect_rate(0.1, confidence_level), axis=1)  # 零配件次品率
    half_product_defects = df_half_products.apply(lambda x: estimate_defect_rate(0.1, confidence_level), axis=1)  # 半成品次品率
    final_product_defect = estimate_defect_rate(0.1, confidence_level)  # 成品次品率

    # 存储每次循环的决策方案和次品率历史
    decisions = {'cycle': [], '零配件检测决策': [], '半成品检测决策': [], '成品检测决策': []}
    defect_rate_history = {'零配件1次品率': [], '零配件2次品率': [], '半成品次品率': [], '成品次品率': []}

    # 循环直到达到最大次数或次品率低于阈值
    while cycle < max_cycles and (
            part_defects.max() > epsilon or half_product_defects.max() > epsilon or final_product_defect > epsilon):
        cycle += 1  # 每次循环

        # 记录每次循环的次品率
        defect_rate_history['零配件1次品率'].append(part_defects[0])
        defect_rate_history['零配件2次品率'].append(part_defects[1])
        defect_rate_history['半成品次品率'].append(half_product_defects.max())
        defect_rate_history['成品次品率'].append(final_product_defect)

        # 零配件检测与成本计算
        part_cost = 0  # 零配件成本
        part_decisions = []  # 零配件检测决策
        for i in range(len(df_parts)):
            # 判断是否进行检测
            if df_parts.loc[i, '检测成本'] < part_defects[i] * df_parts.loc[i, '购买单价']:
                part_cost += df_parts.loc[i, '检测成本']  # 进行检测
                part_decisions.append(1)  # 1 表示检测
            else:
                part_cost += part_defects[i] * df_parts.loc[i, '购买单价']  # 不检测
                part_decisions.append(0)  # 0 表示不检测
        total_cost += part_cost  # 更新总成本

        # 半成品检测与成本计算
        half_product_cost = 0  # 半成品成本
        half_product_decisions = []  # 半成品检测决策
        for i in range(len(df_half_products)):
            # 判断是否进行检测
            if df_half_products.loc[i, '检测成本'] < half_product_defects[i] * (
                    df_half_products.loc[i, '装配成本'] + df_half_products.loc[i, '拆解费用']):
                half_product_cost += df_half_products.loc[i, '检测成本']  # 进行检测
                half_product_decisions.append(1)  # 1 表示检测
            else:
                half_product_cost += half_product_defects[i] * (
                            df_half_products.loc[i, '装配成本'] + df_half_products.loc[i, '拆解费用'])  # 不检测
                half_product_decisions.append(0)  # 0 表示不检测
        total_cost += half_product_cost  # 更新总成本

        # 成品检测与成本计算
        final_product_cost = 0  # 成品成本
        final_product_decision = 0  # 成品检测决策
        if df_final_product.loc[0, '检测成本'] < final_product_defect * (
                df_final_product.loc[0, '装配成本'] + df_final_product.loc[0, '市场售价'] + df_final_product.loc[0, '拆解费用']):
            final_product_cost += df_final_product.loc[0, '检测成本']  # 进行检测
            final_product_decision = 1  # 1 表示检测
        else:
            final_product_cost += final_product_defect * (
                        df_final_product.loc[0, '装配成本'] + df_final_product.loc[0, '市场售价'] +
                        df_final_product.loc[0, '拆解费用'])  # 不检测
            final_product_decision = 0  # 0 表示不检测
        total_cost += final_product_cost  # 更新总成本

        # 保存当前循环的决策
        decisions['cycle'].append(cycle)
        decisions['零配件检测决策'].append(part_decisions)
        decisions['半成品检测决策'].append(half_product_decisions)
        decisions['成品检测决策'].append(final_product_decision)

        # 次品率递减
        part_defects *= (1 - ALPHA_PARTS)  # 更新零配件次品率
        half_product_defects *= (1 - ALPHA_HALF_PRODUCTS)  # 更新半成品次品率
        final_product_defect *= (1 - ALPHA_FINAL_PRODUCT)  # 更新成品次品率

    return total_cost, cycle, decisions, defect_rate_history  # 返回总成本、循环次数、决策和次品率历史


# 进行95%置信度下的总成本计算
total_cost_95, cycle_count_95, decision_results_95, defect_rate_history_95 = calculate_total_cost_with_sampling(
    df_parts, df_half_products, df_final_product, confidence_level=0.95)

# 进行90%置信度下的总成本计算
total_cost_90, cycle_count_90, decision_results_90, defect_rate_history_90 = calculate_total_cost_with_sampling(
    df_parts, df_half_products, df_final_product, confidence_level=0.90)

# 输出95%置信度下的结果
print("\n95%置信度下的决策分析结果：")
print(f"总成本: {total_cost_95:.2f}")  # 输出总成本
print(f"总循环次数: {cycle_count_95}")  # 输出循环次数
for cycle in range(cycle_count_95):
    print(f"第 {decision_results_95['cycle'][cycle]} 次循环：")
    print(f"  零配件检测决策: {decision_results_95['零配件检测决策'][cycle]}")  # 输出零配件检测决策
    print(f"  半成品检测决策: {decision_results_95['半成品检测决策'][cycle]}")  # 输出半成品检测决策
    print(f"  成品检测决策: {decision_results_95['成品检测决策'][cycle]}")  # 输出成品检测决策

# 输出90%置信度下的结果
print("\n90%置信度下的决策分析结果：")
print(f"总成本: {total_cost_90:.2f}")  # 输出总成本
print(f"总循环次数: {cycle_count_90}")  # 输出循环次数
for cycle in range(cycle_count_90):
    print(f"第 {decision_results_90['cycle'][cycle]} 次循环：")
    print(f"  零配件检测决策: {decision_results_90['零配件检测决策'][cycle]}")  # 输出零配件检测决策
    print(f"  半成品检测决策: {decision_results_90['半成品检测决策'][cycle]}")  # 输出半成品检测决策
    print(f"  成品检测决策: {decision_results_90['成品检测决策'][cycle]}")  # 输出成品检测决策

# 输出95%置信度下的次品率历史
print("\n95%置信度下的次品率历史：")
print(f"零配件1次品率历史: {defect_rate_history_95['零配件1次品率']}")  # 输出零配件1次品率历史
print(f"零配件2次品率历史: {defect_rate_history_95['零配件2次品率']}")  # 输出零配件2次品率历史
print(f"半成品次品率历史: {defect_rate_history_95['半成品次品率']}")  # 输出半成品次品率历史
print(f"成品次品率历史: {defect_rate_history_95['成品次品率']}")  # 输出成品次品率历史

# 输出90%置信度下的次品率历史
print("\n90%置信度下的次品率历史：")
print(f"零配件1次品率历史: {defect_rate_history_90['零配件1次品率']}")  # 输出零配件1次品率历史
print(f"零配件2次品率历史: {defect_rate_history_90['零配件2次品率']}")  # 输出零配件2次品率历史
print(f"半成品次品率历史: {defect_rate_history_90['半成品次品率']}")  # 输出半成品次品率历史
print(f"成品次品率历史: {defect_rate_history_90['成品次品率']}")  # 输出成品次品率历史


# 可视化部分：次品率历史折线图
def plot_defect_rate_history(defect_rate_history_95, defect_rate_history_90):
    cycles_95 = range(1, len(defect_rate_history_95['零配件1次品率']) + 1)  # 95%置信度的循环次数
    cycles_90 = range(1, len(defect_rate_history_90['零配件1次品率']) + 1)  # 90%置信度的循环次数
    # 绘制 95% 置信度下的次品率历史
    plt.figure(figsize=(10, 6))
    plt.plot(cycles_95, defect_rate_history_95['零配件1次品率'], label='零配件1次品率 (95%)', color='r', marker='o')  # 零配件1次品率
    plt.plot(cycles_95, defect_rate_history_95['零配件2次品率'], label='零配件2次品率 (95%)', color='g', marker='o')  # 零配件2次品率
    plt.plot(cycles_95, defect_rate_history_95['半成品次品率'], label='半成品次品率 (95%)', color='b', marker='o')  # 半成品次品率
    plt.plot(cycles_95, defect_rate_history_95['成品次品率'], label='成品次品率 (95%)', color='c', marker='o')  # 成品次品率

    plt.xlabel('循环次数')  # X轴标签
    plt.ylabel('次品率')  # Y轴标签
    plt.title('95% 置信度下的次品率历史')  # 图表标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图表

    # 绘制 90% 置信度下的次品率历史
    plt.figure(figsize=(10, 6))
    plt.plot(cycles_90, defect_rate_history_90['零配件1次品率'], label='零配件1次品率 (90%)', color='r', marker='o')  # 零配件1次品率
    plt.plot(cycles_90, defect_rate_history_90['零配件2次品率'], label='零配件2次品率 (90%)', color='g', marker='o')  # 零配件2次品率
    plt.plot(cycles_90, defect_rate_history_90['半成品次品率'], label='半成品次品率 (90%)', color='b', marker='o')  # 半成品次品率
    plt.plot(cycles_90, defect_rate_history_90['成品次品率'], label='成品次品率 (90%)', color='c', marker='o')  # 成品次品率

    plt.xlabel('循环次数')  # X轴标签
    plt.ylabel('次品率')  # Y轴标签
    plt.title('90% 置信度下的次品率历史')  # 图表标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图表


# 调用可视化函数
plot_defect_rate_history(defect_rate_history_95, defect_rate_history_90)  # 绘制次品率历史图
