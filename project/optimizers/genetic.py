# import random
from deap import base, creator, tools, algorithms
import numpy as np

# DOCS:
#   Tools: https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.initRepeat
#   Algorithms: https://deap.readthedocs.io/en/master/api/algo.html

def normalize_last_axis(array):
    swapped = np.swapaxes(array, 0, len(array.shape)-1)
    normalized_swapped = swapped/np.linalg.norm(swapped, axis=0)
    return np.swapaxes(normalized_swapped, 0, len(array.shape)-1)

def normalize_total(array):
    flat = array.flatten()
    return array/np.linalg.norm(flat)

def evalRelMax(individual, shape):
    individual = np.array(individual)
    individual = individual.reshape(shape)
    individual = normalize_total(individual)
    return max(individual.flatten()),

evalRelMax.input_shape = (3,3,2)

def myMutate(individual):
    for i in range(len(individual)):
        individual[i] = individual[i]+np.random.random()
    return individual,

def default_reward_table(shape):
    '''
        call r_rable.cast(default_reward_table(shape))
    '''
    # prod = lambda s: reduce(lambda a, b: a*b, s)
    list_to_rewtable = lambda x: np.array(x).reshape(shape)
    init_rew_list = [-0.1/np.prod(shape[:-1])]*(np.prod(shape)-shape[-1])+[1]*shape[-1]
    return list_to_rewtable(init_rew_list)


class Genetic(object):

    def __init__(self, *args, value_bound=10,  **kwargs):
        super(Genetic, self).__init__(*args, **kwargs)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        value_bound = 10
        self.toolbox = base.Toolbox()
        # self.toolbox.register("attr_bool", lambda: 2*value_bound*random.random()-value_bound)
        # self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, n=np.prod(shape))


        self.toolbox.register("mate", tools.cxTwoPoint)
        # self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
        self.toolbox.register("mutate", myMutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)


    def optimize(self, objective, initializer=lambda: default_reward_table(objective.input_shape), ngen=10):


        self.toolbox.register("evaluate", objective, shape=objective.input_shape)

        # self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, n=np.prod(objective.input_shape))
        self.toolbox.register("individual", tools.initIterate, creator.Individual, initializer)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        pop = self.toolbox.population(n=50)


        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.8, ngen=ngen,
                                            stats=self.stats, halloffame=self.hof, verbose=True)

        return pop, logbook, self.hof

if __name__ == "__main__":
    pop, log, hof = Genetic().optimize(evalRelMax)
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

    import matplotlib.pyplot as plt
    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    # plt.show()

# if __name__ == '__main__':
#     print(Genetic(evalRelMax, (3,4,2)).optimize())
