import random
from deap import base, creator, tools, algorithms
import numpy as np

class Genetic(object):
    """docstring for Genetic."""

    def __init__(self, objective, ind_init, *args, value_bound=10,  **kwargs):
        super(Genetic, self).__init__(*args, **kwargs)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", lambda: 2*value_bound*random.random()-value_bound)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=reduce(lambda a,b : a*b, shape))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)



    def optimize(self, init_population, num_generations=10):
        pop = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        pop, logbook = algorithms.eaSimple(pop, toolbox,
                                            cxpb=0.5, mutpb=0.2, ngen=10,
                                            stats=stats, halloffame=hof, verbose=True)

        return pop, logbook, hof

def normalize_last_axis(array):
    swapped = np.swapaxes(array, 0, len(array.shape)-1)
    normalized_swapped = swapped/np.linalg.norm(swapped, axis=0)
    return np.swapaxes(normalized_swapped, 0, len(array.shape)-1)

def normalize_total(array):
    flat = array.flatten()
    return array/np.linalg.norm(flat)

def evalRelMax(individual):
    individual = np.array(individual)
    individual = individual.reshape(shape)
    individual = normalize_total(individual)
    return max(individual.flatten()),
