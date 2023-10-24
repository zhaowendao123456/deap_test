from deap import base, creator, tools
import random
# 创建两个序列编码个体
random.seed(42) # 保证结果可复现
IND_SIZE = 8
creator.create('FitnessMin', base.Fitness, weights=(-1.0, ))
creator.create('Individual', list, fitness = creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register('Indices', random.sample, range(IND_SIZE), IND_SIZE)
toolbox.register('Individual', tools.initIterate, creator.Individual, toolbox.Indices)

ind1, ind2 = [toolbox.Individual() for _ in range(2)]
print(ind1, '\n', ind2)

# 单点交叉
child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxOnePoint(child1, child2)
print("单点交叉")
print(child1, '\n', child2)

# 两点交叉
child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxTwoPoint(child1, child2)
print("两点交叉")
print(child1, '\n', child2)

# 均匀交叉
child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxUniform(child1, child2, 0.5)
print("均匀交叉")
print(child1, '\n', child2)

# 部分匹配交叉
child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxPartialyMatched(child1, child2)
print("部分匹配交叉")
print(child1, '\n', child2)

# 有序交叉
child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxOrdered(child1, child2)
print("有序交叉")
print(child1, '\n', child2)

# 混乱单点交叉
child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxMessyOnePoint(child1, child2)
print("混乱单点交叉")
print(child1, '\n', child2)

