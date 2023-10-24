from deap import base, creator, tools
import numpy as np

# 定义问题
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))  # 优化目标：单变量，求最小值
creator.create('Individual', list, fitness=creator.FitnessMin)  # 创建Individual类，继承list

# 生成个体
IND_SIZE = 5
toolbox = base.Toolbox()
toolbox.register('Attr_float', np.random.rand)
toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Attr_float, n=IND_SIZE)

# 生成初始族群
N_POP = 10
toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)
pop = toolbox.Population(n=N_POP)

# 定义评价函数
def evaluate(individual):
    return sum(individual),  # 注意这个逗号，即使是单变量优化问题，也需要返回tuple

# 评价初始族群
toolbox.register('Evaluate', evaluate)
fitnesses = map(toolbox.Evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# 选择方式1：锦标赛选择
toolbox.register('TourSel', tools.selTournament, tournsize=2)  # 注册Tournsize为2的锦标赛选择
selectedTour = toolbox.TourSel(pop, 5)  # 选择5个个体
print('锦标赛选择结果：')
for ind in selectedTour:
    print(ind)
    print(ind.fitness.values)

# 选择方式2: 轮盘赌选择
toolbox.register('RoulSel', tools.selRoulette)
selectedRoul = toolbox.RoulSel(pop, 5)
print('轮盘赌选择结果：')
for ind in selectedRoul:
    print(ind)
    print(ind.fitness.values)

# 选择方式3: 随机普遍抽样选择
toolbox.register('StoSel', tools.selStochasticUniversalSampling)
selectedSto = toolbox.StoSel(pop, 5)
print('随机普遍抽样选择结果：')
for ind in selectedSto:
    print(ind)
    print(ind.fitness.values)