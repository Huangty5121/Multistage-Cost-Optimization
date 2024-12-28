import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib绘图时可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体，以支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 使得坐标轴负号显示正常

# 输入数据
cases = ['情况1', '情况2', '情况3', '情况4', '情况5', '情况6']  # 定义情况标签
total_costs = [30.8150, 47.8705, 30.8150, 40.7270, 42.2620, 5.5700]  # 定义每种情况的总成本
cycles = [2, 3, 2, 3, 3, 2]  # 定义每种情况的循环次数

def set_plot_params(ax, xlabel, ylabel, title):
    # 设置图表的X轴标签、Y轴标签和标题
    ax.set_xlabel(xlabel, fontsize=14)  # 设置X轴标签
    ax.set_ylabel(ylabel, fontsize=14)  # 设置Y轴标签
    ax.set_title(title, fontsize=16)  # 设置图表标题

# 设置图表尺寸
fig, ax1 = plt.subplots(figsize=(10, 6))  # 创建一个10x6英寸的图表

# 绘制总成本柱状图
bars = ax1.bar(cases, total_costs, color='steelblue', label='总成本')  # 绘制柱状图，颜色为钢蓝色
set_plot_params(ax1, '情况', '总成本', '各情况的总成本与循环次数')  # 设置图表参数

# 在每个柱状图上方添加数值标签
for bar in bars:
    yval = bar.get_height()  # 获取柱子的高度
    ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=12)  # 在柱子上方添加数值标签

# 创建第二个Y轴用于绘制循环次数折线图
ax2 = ax1.twinx()  # 创建一个共享X轴的第二个Y轴
ax2.plot(cases, cycles, color='red', marker='o', label='循环次数')  # 绘制折线图，颜色为红色
ax2.set_ylabel('循环次数', fontsize=14)  # 设置第二个Y轴的标签

# 添加图例
ax1.legend(loc='upper left')  # 在左上角添加第一个图例
ax2.legend(loc='upper right')  # 在右上角添加第二个图例

# 显示图表
plt.show()  # 显示绘制的图表
