import operator
import math
import random
from deap import base,creator,tools,gp

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# 定义一个新的原始集合
pset = gp.PrimitiveSet("MAIN", arity=1)

# 添加基本操作到原始集合
pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.sub, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addPrimitive(operator.neg, arity=1)
pset.addPrimitive(math.cos, arity=1)
pset.addPrimitive(math.sin, arity=1)

# 添加终端到原始集合
pset.addTerminal(1)
pset.addTerminal(-1)

# 创建工具箱
toolbox = base.Toolbox()

# 注册个体和种群的初始化函数到工具箱
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册评估函数到工具箱
def evalSymbReg(individual):
    # 将个体转换为可执行函数
    func = gp.compile(expr=individual,pset=pset)
    # 计算并返回个体的适应度值
    return sum((func(x) - x**3 - x**2 - x)**2 for x in range(-10, 10)),

toolbox.register("evaluate", evalSymbReg)

# 注册遗传操作到工具箱
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# 设置遗传操作的概率
probab_crossover = 0.5
probab_mutate = 0.2

# 创建种群并初始化其个体
pop = toolbox.population(n=300)

# 进行遗传算法的主循环
for g in range(100):
    # 选择下一代个体
    offspring = toolbox.select(pop, len(pop))
    # 克隆选中的个体，以防止父代和子代之间的引用
    offspring = list(map(toolbox.clone, offspring))

    # 应用交叉和突变操作
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < probab_crossover:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < probab_mutate:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 评估新生成的个体，并更新种群
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

# 打印最优个体和其适应度值
best_ind = tools.selBest(pop,k=1)[0]
print(f'最佳个体为 {best_ind}，适应度值为 {best_ind.fitness.values[0]}')
