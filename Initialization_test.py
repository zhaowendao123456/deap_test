from deap import base, creator, tools
from scipy.stats import bernoulli
import random

# 定义问题
creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) # 单目标，最小化
creator.create('Individual', list, fitness = creator.FitnessMin)

# 生成个体
IND_SIZE = 3
toolbox = base.Toolbox()
toolbox.register('Attr_float', random.random)
toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Attr_float, n=IND_SIZE)

# 生成初始族群
N_POP = 5
toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)
pop1 = toolbox.Population(n = N_POP)
print(pop1)


