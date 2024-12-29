import pandas as pd
from deap import base, creator, tools, algorithms
import random

# 定义表 1 中的情况数据
data = {
    # 各种零配件的次品率、购买单价、检测成本等
    '零配件1次品率': [0.10, 0.20, 0.10, 0.20, 0.10, 0.05],
    '零配件1购买单价': [4, 4, 4, 4, 4, 4],
    '零配件1检测成本': [2, 2, 2, 1, 8, 2],
    '零配件2次品率': [0.10, 0.20, 0.10, 0.20, 0.20, 0.05],
    '零配件2购买单价': [18, 18, 18, 18, 18, 18],
    '零配件2检测成本': [3, 3, 3, 1, 1, 3],
    '成品次品率': [0.10, 0.20, 0.10, 0.20, 0.10, 0.05],
    '成品装配成本': [6, 6, 6, 6, 6, 6],
    '成品检测成本': [3, 3, 3, 2, 2, 3],
    '市场售价': [56, 56, 56, 56, 56, 56],
    '调换损失': [6, 6, 30, 30, 10, 10],
    '拆解费用': [5, 5, 5, 5, 5, 40]
}

# 转换为DataFrame格式
df = pd.DataFrame(data)

# 设定终止条件
MAX_CYCLES = 10  # 最多循环10次
EPSILON = 0.015  # 次品率阈值为1.5%

# 模型中的次品率下降比例
ALPHA_1 = 0.65  # 零配件1次品率递减比例
ALPHA_2 = 0.65  # 零配件2次品率递减比例
ALPHA_PRODUCT = 0.65  # 成品次品率递减比例

# 决策函数，输入为各项参数，输出为是否检测、是否拆解、总成本
def decision_analysis(row):
    # 初始变量
    p1_defect_rate = row['零配件1次品率']  # 零配件1的次品率
    p2_defect_rate = row['零配件2次品率']  # 零配件2的次品率
    product_defect_rate = row['成品次品率']  # 成品的次品率

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

    total_cost = 0  # 初始化总成本
    cycle = 0  # 循环计数

    # 开始循环，直到满足终止条件
    while cycle < MAX_CYCLES and (
            p1_defect_rate > EPSILON or p2_defect_rate > EPSILON or product_defect_rate > EPSILON):
        cycle += 1  # 每次循环

        # 决策规则：是否检测零配件1，零配件2
        test_c1 = c1_test_cost < (p1_defect_rate * c1_price)  # 判断是否检测零配件1
        test_c2 = c2_test_cost < (p2_defect_rate * c2_price)  # 判断是否检测零配件2
        test_product = product_test_cost < (product_defect_rate * (assembly_cost + sale_price))  # 判断是否检测成品
        disassemble = disassemble_cost < replace_loss  # 判断是否拆解

        # 每次循环的成本计算
        cycle_cost = (test_c1 * c1_test_cost + (1 - test_c1) * p1_defect_rate * c1_price +
                      test_c2 * c2_test_cost + (1 - test_c2) * p2_defect_rate * c2_price +
                      test_product * product_test_cost + (1 - test_product) * product_defect_rate * (assembly_cost + sale_price))

        # 如果拆解，增加拆解成本和重新检测的成本
        if disassemble:
            cycle_cost += (disassemble_cost + 
                           c1_test_cost * p1_defect_rate + 
                           c2_test_cost * p2_defect_rate + 
                           assembly_cost)  # 重新装配的成本

        total_cost += cycle_cost  # 将每次循环的成本加入总成本

        # 次品率递减
        p1_defect_rate *= (1 - ALPHA_1)  # 更新零配件1的次品率
        p2_defect_rate *= (1 - ALPHA_2)  # 更新零配件2的次品率
        product_defect_rate *= (1 - ALPHA_PRODUCT)  # 更新成品的次品率

    return {
        '检测零配件1': test_c1,  # 返回检测零配件1的决策
        '检测零配件2': test_c2,  # 返回检测零配件2的决策
        '检测成品': test_product,  # 返回检测成品的决策
        '拆解': disassemble,  # 返回拆解的决策
        '总成本': total_cost,  # 返回总成本
        '循环次数': cycle  # 记录实际循环次数
    }

# 对每种情况执行决策分析
decisions = df.apply(decision_analysis, axis=1)

# 将决策结果转换为DataFrame格式
decision_results = pd.DataFrame(decisions.tolist())

# 显示结果
print(decision_results)

# 定义适应度函数
def evaluate(individual):
    # 将个体解码为参数
    p1_defect_rate = individual[0]  # 零配件1的次品率
    p2_defect_rate = individual[1]  # 零配件2的次品率
    product_defect_rate = individual[2]  # 成品的次品率
    
    total_cost = 0  # 初始化总成本
    cycle = 0  # 循环计数

    while cycle < MAX_CYCLES and (
            p1_defect_rate > EPSILON or p2_defect_rate > EPSILON or product_defect_rate > EPSILON):
        cycle += 1
        # 计算成本（与之前相同）
        # ... 计算逻辑 ...

        # 次品率递减
        p1_defect_rate *= (1 - ALPHA_1)  # 更新零配件1的次品率
        p2_defect_rate *= (1 - ALPHA_2)  # 更新零配件2的次品率
        product_defect_rate *= (1 - ALPHA_PRODUCT)  # 更新成品的次品率

    return total_cost,  # 返回总成本

# 遗传算法设置
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 定义适应度最小化
creator.create("Individual", list, fitness=creator.FitnessMin)  # 定义个体

toolbox = base.Toolbox()  # 创建工具箱
toolbox.register("p1_defect_rate", random.uniform, 0, 1)  # 注册零配件1的次品率生成
toolbox.register("p2_defect_rate", random.uniform, 0, 1)  # 注册零配件2的次品率生成
toolbox.register("product_defect_rate", random.uniform, 0, 1)  # 注册成品的次品率生成

# 使用 initIterate 创建个体
toolbox.register("individual", tools.initIterate, creator.Individual, 
                 lambda: [toolbox.p1_defect_rate(), toolbox.p2_defect_rate(), toolbox.product_defect_rate()])

toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # 注册种群
toolbox.register("evaluate", evaluate)  # 注册评估函数
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 注册交叉操作
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # 注册变异操作
toolbox.register("select", tools.selTournament, tournsize=3)  # 注册选择操作

# 运行遗传算法
population = toolbox.population(n=50)  # 创建种群
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)  # 运行遗传算法

# 获取最佳个体
best_individual = tools.selBest(population, k=1)[0]  # 选择最佳个体