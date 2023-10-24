from deap import base, creator, tools
import random
# 创建一个实数编码个体
random.seed(42) # 保证结果可复现
IND_SIZE = 5
creator.create('FitnessMin', base.Fitness, weights=(-1.0, ))
creator.create('Individual', list, fitness = creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register('Attr_float', random.random)
toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Attr_float, IND_SIZE)

ind1 = toolbox.Individual()
print(ind1)

# 高斯突变
mutant = toolbox.clone(ind1)
tools.mutGaussian(mutant, 3, 0.1, 1)
print("高斯突变")
print(mutant)

# 乱序突变
mutant = toolbox.clone(ind1)
tools.mutShuffleIndexes(mutant, 0.5)
print("乱序突变")
print(mutant)

# 有界多项式突变
mutant = toolbox.clone(ind1)
tools.mutPolynomialBounded(mutant, 20, 0, 1, 0.5)
print("有界多项式突变")
print(mutant)

# 均匀整数突变
mutant = toolbox.clone(ind1)
tools.mutUniformInt(mutant, 1, 5, 0.5)
print("均匀整数突变")
print(mutant)

