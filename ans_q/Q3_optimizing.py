import pandas as pd

# 定义问题中的数据
data = {
    '零配件': ['零配件1', '零配件2', '零配件3', '零配件4', '零配件5', '零配件6', '零配件7', '零配件8'],
    '次品率': [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    '购买单价': [2, 8, 12, 2, 8, 12, 8, 12],
    '检测成本': [1, 1, 2, 1, 1, 2, 1, 2]
}

half_products = {
    '半成品': ['半成品1', '半成品2', '半成品3'],
    '次品率': [0.10, 0.10, 0.10],
    '装配成本': [8, 8, 8],
    '检测成本': [4, 4, 4],
    '拆解费用': [6, 6, 6]
}

final_product = {
    '成品': ['成品'],
    '次品率': [0.10],
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
EPSILON = 0.015  # 次品率阈值为2%

# 定义次品率递减比例
ALPHA_PARTS = 0.65  # 零配件次品率递减比例
ALPHA_HALF_PRODUCTS = 0.65  # 半成品次品率递减比例
ALPHA_FINAL_PRODUCT = 0.65  # 成品次品率递减比例

# 定义总成本计算函数，并输出具体决策方案
def calculate_total_cost_and_decisions(df_parts, df_half_products, df_final_product, epsilon=EPSILON,
                                       max_cycles=MAX_CYCLES):
    total_cost = 0  # 初始化总成本
    cycle = 0  # 循环计数

    # 初始化次品率
    part_defects = df_parts['次品率'].copy()  # 零配件次品率
    half_product_defects = df_half_products['次品率'].copy()  # 半成品次品率
    final_product_defect = df_final_product['次品率'][0]  # 成品次品率

    # 存储每次循环的决策方案
    decisions = {'cycle': [], '零配件检测决策': [], '半成品检测决策': [], '成品检测决策': []}

    # 循环直到达到最大次数或所有次品率低于阈值
    while cycle < max_cycles and (
            part_defects.max() > epsilon or half_product_defects.max() > epsilon or final_product_defect > epsilon):
        cycle += 1  # 每次循环

        # 零配件检测与成本计算
        part_cost = 0  # 零配件成本初始化
        part_decisions = []  # 零配件决策初始化
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
        half_product_cost = 0  # 半成品成本初始化
        half_product_decisions = []  # 半成品决策初始化
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
        final_product_cost = 0  # 成品成本初始化
        final_product_decision = 0  # 成品决策初始化
        if df_final_product.loc[0, '检测成本'] < final_product_defect * (
                df_final_product.loc[0, '装配成本'] + df_final_product.loc[0, '市场售价'] + df_final_product.loc[
            0, '拆解费用']):
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

    return total_cost, cycle, decisions  # 返回总成本、循环次数和决策方案


# 进行总成本计算，并输出决策
total_cost, cycle_count, decision_results = calculate_total_cost_and_decisions(df_parts, df_half_products,
                                                                               df_final_product)

# 输出结果
print(f"总成本: {total_cost:.2f}")  # 打印总成本
print(f"总循环次数: {cycle_count}")  # 打印循环次数
print("具体决策方案如下：")  # 打印决策方案
for cycle in range(cycle_count):
    print(f"第 {decision_results['cycle'][cycle]} 次循环：")
    print(f"  零配件检测决策: {decision_results['零配件检测决策'][cycle]}")  # 打印零配件决策
    print(f"  半成品检测决策: {decision_results['半成品检测决策'][cycle]}")  # 打印半成品决策
    print(f"  成品检测决策: {decision_results['成品检测决策'][cycle]}")  # 打印成品决策

# 导入绘图库
import matplotlib.pyplot as plt
import numpy as np
# 设置matplotlib绘图时可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 使得坐标轴负号显示正常

# 提取决策结果用于可视化
part_decisions_matrix = np.array(decision_results['零配件检测决策'])  # 零配件决策矩阵
half_product_decisions_matrix = np.array(decision_results['半成品检测决策'])  # 半成品决策矩阵
final_product_decisions = decision_results['成品检测决策']  # 成品决策

# 设置循环次数标签
cycles_labels = [f"循环 {i+1}" for i in range(cycle_count)]  # 循环次数标签

# 可视化零配件检测决策 (每个循环中零配件的检测决策 0/1)
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(part_decisions_matrix.T, cmap='Blues', aspect='auto')  # 绘制热图

# 设置标题和坐标标签
ax.set_title("零配件检测决策（0=不检测, 1=检测）")  # 设置标题
ax.set_xticks(np.arange(len(cycles_labels)))  # 设置x轴刻度
ax.set_xticklabels(cycles_labels)  # 设置x轴刻度标签
ax.set_yticks(np.arange(len(df_parts)))  # 设置y轴刻度
ax.set_yticklabels(df_parts['零配件'])  # 设置y轴刻度标签

# 标注每个格子的数值
for i in range(len(df_parts)):
    for j in range(cycle_count):
        text = ax.text(j, i, part_decisions_matrix[j, i],
                       ha="center", va="center", color="black")  # 在每个格子中添加文本

# 设置色条
fig.colorbar(im)  # 添加色条

plt.show()  # 显示图形

# 可视化半成品和成品的检测决策
fig, ax2 = plt.subplots(figsize=(10, 6))
x = np.arange(cycle_count)  # 循环次数
width = 0.35  # 条形图的宽度

# 绘制半成品检测决策的条形图
rects1 = ax2.bar(x - width/2, np.sum(half_product_decisions_matrix, axis=1), width, label='半成品检测决策')
# 绘制成品检测决策的折线图
rects2 = ax2.plot(x, final_product_decisions, color='orange', marker='o', label='成品检测决策')

# 添加文本标签
ax2.set_xlabel('循环次数')  # x轴标签
ax2.set_ylabel('决策结果 (0=不检测, 1=检测)')  # y轴标签
ax2.set_title('半成品和成品检测决策')  # 图形标题
ax2.set_xticks(x)  # 设置x轴刻度
ax2.set_xticklabels(cycles_labels)  # 设置x轴刻度标签
ax2.legend()  # 添加图例

plt.show()  # 显示图形

# 绘制总成本的变化趋势
fig, ax3 = plt.subplots(figsize=(10, 6))
cost_per_cycle = [calculate_total_cost_and_decisions(df_parts, df_half_products, df_final_product, max_cycles=i+1)[0] for i in range(cycle_count)]  # 计算每个循环的总成本
ax3.plot(cycles_labels, cost_per_cycle, marker='o', color='green')  # 绘制总成本变化趋势图

ax3.set_title('总成本随循环次数变化趋势')  # 图形标题
ax3.set_xlabel('循环次数')  # x轴标签
ax3.set_ylabel('总成本')  # y轴标签
plt.grid(True)  # 添加网格
plt.show()  # 显示图形
