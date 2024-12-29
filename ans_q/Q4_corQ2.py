# 导入必要的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
from scipy.stats import norm  # 用于统计分布

# 定义表 1 中的情况数据
data = {
    '零配件1次品率': [0.10, 0.20, 0.10, 0.20, 0.10, 0.05],  # 零配件1的次品率
    '零配件1购买单价': [4, 4, 4, 4, 4, 4],  # 零配件1的购买单价
    '零配件1检测成本': [2, 2, 2, 1, 8, 2],  # 零配件1的检测成本
    '零配件2次品率': [0.10, 0.20, 0.10, 0.20, 0.20, 0.05],  # 零配件2的次品率
    '零配件2购买单价': [18, 18, 18, 18, 18, 18],  # 零配件2的购买单价
    '零配件2检测成本': [3, 3, 3, 1, 1, 3],  # 零配件2的检测成本
    '成品次品率': [0.10, 0.20, 0.10, 0.20, 0.10, 0.05],  # 成品的次品率
    '成品装配成本': [6, 6, 6, 6, 6, 6],  # 成品的装配成本
    '成品检测成本': [3, 3, 3, 2, 2, 3],  # 成品的检测成本
    '市场售价': [56, 56, 56, 56, 56, 56],  # 成品的市场售价
    '调换损失': [6, 6, 30, 30, 10, 10],  # 调换损失
    '拆解费用': [5, 5, 5, 5, 5, 40]  # 拆解费用
}

# 转换为DataFrame格式
df = pd.DataFrame(data)  # 将数据转换为Pandas DataFrame格式，便于后续处理

# 设定终止条件
MAX_CYCLES = 10  # 最多循环10次
EPSILON = 0.015  # 次品率阈值为1.5%

# 模型中的次品率下降比例
ALPHA_1 = 0.65  # 零配件1次品率递减比例
ALPHA_2 = 0.65  # 零配件2次品率递减比例
ALPHA_PRODUCT = 0.65  # 成品次品率递减比例

# 抽样检测中的参数
Z_ALPHA_95 = 1.96  # 95% 置信度下的 Z 值
Z_ALPHA_90 = 1.645  # 90% 置信度下的 Z 值
ERROR_MARGIN = 0.02  # 允许的误差

# 基于抽样检测的次品率估计函数
def estimate_defect_rate(p_estimate, confidence_level, error_margin=ERROR_MARGIN):
    z_value = norm.ppf((1 + confidence_level) / 2)  # 计算Z值
    sample_size = (z_value ** 2 * p_estimate * (1 - p_estimate)) / (error_margin ** 2)  # 计算样本大小
    # 从样本估计次品率，样本次品率为给定的 p_estimate，但添加一些随机波动
    sample_defect_rate = np.random.binomial(int(sample_size), p_estimate) / sample_size  # 计算样本次品率
    return sample_defect_rate  # 返回估计的次品率

# 决策函数，输入为各项参数，输出为是否检测、是否拆解、总成本以及抽样后的次品率
def decision_analysis(row, confidence_scenario=95):
    # 选择不同的置信水平
    if confidence_scenario == 95:
        confidence_level = 0.95  # 设定置信水平为95%
        z_value = Z_ALPHA_95  # 使用95%置信度的Z值
    elif confidence_scenario == 90:
        confidence_level = 0.90  # 设定置信水平为90%
        z_value = Z_ALPHA_90  # 使用90%置信度的Z值

    # 初始变量，通过抽样检测估计次品率
    p1_defect_rate = estimate_defect_rate(row['零配件1次品率'], confidence_level)  # 零配件1的次品率
    p2_defect_rate = estimate_defect_rate(row['零配件2次品率'], confidence_level)  # 零配件2的次品率
    product_defect_rate = estimate_defect_rate(row['成品次品率'], confidence_level)  # 成品的次品率

    # 成本参数
    c1_price = row['零配件1购买单价']  # 零配件1的购买单价
    c2_price = row['零配件2购买单价']  # 零配件2的购买单价
    assembly_cost = row['成品装配成本']  # 成品的装配成本
    sale_price = row['市场售价']  # 成品的市场售价
    replace_loss = row['调换损失']  # 调换损失
    disassemble_cost = row['拆解费用']  # 拆解费用

    # 检测成本
    c1_test_cost = row['零配件1检测成本']  # 零配件1的检测成本
    c2_test_cost = row['零配件2检测成本']  # 零配件2的检测成本
    product_test_cost = row['成品检测成本']  # 成品的检测成本

    # 重新装配和拆解后的检测成本
    rework_test_cost_1 = c1_test_cost  # 假设拆解后的零配件1检测成本和原检测成本相同
    rework_test_cost_2 = c2_test_cost  # 拆解后的零配件2检测成本
    rework_assembly_cost = assembly_cost  # 重新装配的成本和原装配成本一致

    total_cost = 0  # 初始化总成本
    cycle = 0  # 循环计数

    # 存储每次循环的次品率
    defect_rate_history = {
        '零配件1次品率': [],  # 存储零配件1的次品率历史
        '零配件2次品率': [],  # 存储零配件2的次品率历史
        '成品次品率': []  # 存储成品的次品率历史
    }

    # 开始循环，直到满足终止条件
    while cycle < MAX_CYCLES and (
            p1_defect_rate > EPSILON or p2_defect_rate > EPSILON or product_defect_rate > EPSILON):
        cycle += 1  # 每次循环

        # 记录每次循环的次品率
        defect_rate_history['零配件1次品率'].append(p1_defect_rate)
        defect_rate_history['零配件2次品率'].append(p2_defect_rate)
        defect_rate_history['成品次品率'].append(product_defect_rate)

        # 决策规则：是否检测零配件1，零配件2
        test_c1 = c1_test_cost < (p1_defect_rate * c1_price)  # 判断是否检测零配件1
        test_c2 = c2_test_cost < (p2_defect_rate * c2_price)  # 判断是否检测零配件2

        # 是否检测成品
        test_product = product_test_cost < (product_defect_rate * (assembly_cost + sale_price))  # 判断是否检测成品

        # 是否拆解
        disassemble = disassemble_cost < replace_loss  # 判断是否拆解

        # 每次循环的成本计算
        cycle_cost = 0  # 初始化循环成本
        cycle_cost += test_c1 * c1_test_cost + (1 - test_c1) * p1_defect_rate * c1_price  # 零配件1的成本
        cycle_cost += test_c2 * c2_test_cost + (1 - test_c2) * p2_defect_rate * c2_price  # 零配件2的成本
        cycle_cost += test_product * product_test_cost + (1 - test_product) * product_defect_rate * (
                    assembly_cost + sale_price)  # 成品的成本

        # 如果拆解，增加拆解成本和重新检测的成本
        if disassemble:
            cycle_cost += disassemble_cost  # 拆解成本
            # 重新检测拆解后的零配件成本
            cycle_cost += rework_test_cost_1 * p1_defect_rate  # 零配件1的重新检测成本
            cycle_cost += rework_test_cost_2 * p2_defect_rate  # 零配件2的重新检测成本
            cycle_cost += rework_assembly_cost  # 重新装配的成本

        total_cost += cycle_cost  # 将每次循环的成本加入总成本

        # 次品率递减
        p1_defect_rate *= (1 - ALPHA_1)  # 更新零配件1的次品率
        p2_defect_rate *= (1 - ALPHA_2)  # 更新零配件2的次品率
        product_defect_rate *= (1 - ALPHA_PRODUCT)  # 更新成品的次品率

    return {
        '检测零配件1': test_c1,  # 返回是否检测零配件1
        '检测零配件2': test_c2,  # 返回是否检测零配件2
        '检测成品': test_product,  # 返回是否检测成品
        '拆解': disassemble,  # 返回是否拆解
        '总成本': total_cost,  # 返回总成本
        '循环次数': cycle,  # 记录实际循环次数
        '零配件1次品率历史': defect_rate_history['零配件1次品率'],  # 返回零配件1的次品率历史
        '零配件2次品率历史': defect_rate_history['零配件2次品率'],  # 返回零配件2的次品率历史
        '成品次品率历史': defect_rate_history['成品次品率']  # 返回成品的次品率历史
    }

# 对每种情况执行决策分析
# 使用95%置信度的结果
decisions_95 = df.apply(decision_analysis, axis=1, confidence_scenario=95)  # 应用决策分析函数
# 使用90%置信度的结果
decisions_90 = df.apply(decision_analysis, axis=1, confidence_scenario=90)  # 应用决策分析函数

# 将决策结果转换为DataFrame格式
decision_results_95 = pd.DataFrame(decisions_95.tolist())  # 将95%置信度的结果转换为DataFrame
decision_results_90 = pd.DataFrame(decisions_90.tolist())  # 将90%置信度的结果转换为DataFrame

# 显示95%置信度结果
print("\n95%置信度下的决策分析结果：")
print(decision_results_95[['检测零配件1', '检测零配件2', '检测成品', '拆解', '总成本', '循环次数']])  # 打印95%置信度的决策结果

# 显示90%置信度结果
print("\n90%置信度下的决策分析结果：")
print(decision_results_90[['检测零配件1', '检测零配件2', '检测成品', '拆解', '总成本', '循环次数']])  # 打印90%置信度的决策结果

# 提取每种情况的次品率历史，并以表格形式输出
defect_rate_results_95 = pd.DataFrame({
    '零配件1次品率历史': decision_results_95['零配件1次品率历史'],  # 零配件1的次品率历史
    '零配件2次品率历史': decision_results_95['零配件2次品率历史'],  # 零配件2的次品率历史
    '成品次品率历史': decision_results_95['成品次品率历史']  # 成品的次品率历史
})

defect_rate_results_90 = pd.DataFrame({
    '零配件1次品率历史': decision_results_90['零配件1次品率历史'],  # 零配件1的次品率历史
    '零配件2次品率历史': decision_results_90['零配件2次品率历史'],  # 零配件2的次品率历史
    '成品次品率历史': decision_results_90['成品次品率历史']  # 成品的次品率历史
})

# 输出次品率历史
print("\n95%置信度下的次品率历史：")
for index, row in defect_rate_results_95.iterrows():  # 遍历95%置信度的次品率历史
    print(f"\n情况 {index + 1} 的次品率历史：")  # 打印情况编号
    print(f"零配件1次品率历史: {row['零配件1次品率历史']}")  # 打印零配件1的次品率历史
    print(f"零配件2次品率历史: {row['零配件2次品率历史']}")  # 打印零配件2的次品率历史
    print(f"成品次品率历史: {row['成品次品率历史']}")  # 打印成品的次品率历史

print("\n90%置信度下的次品率历史：")
for index, row in defect_rate_results_90.iterrows():  # 遍历90%置信度的次品率历史
    print(f"\n情况 {index + 1} 的次品率历史：")  # 打印情况编号
    print(f"零配件1次品率历史: {row['零配件1次品率历史']}")  # 打印零配件1的次品率历史
    print(f"零配件2次品率历史: {row['零配件2次品率历史']}")  # 打印零配件2的次品率历史
    print(f"成品次品率历史: {row['成品次品率历史']}")  # 打印成品的次品率历史

