import random
import math


#========================================
#                CONSTS
#========================================
POPULATION_SIZE = 1000
P_CROSSOVER = 0.9
P_MUTATION = 0.3
MAX_GENERATIONS = 300
ELITISM_COUNT = 20

INT_ACCURACY = 8
MANTIC_ACCURACY = 20
ONE_MAX_LENGTH = INT_ACCURACY + MANTIC_ACCURACY + 1

OUTPUT_FREQUENCE = 20
BORDER = [[-10, 10],[-10, 10]]


#========================================
#                FUNCTION
#========================================
FUNCTION = lambda x, y: -(math.fabs(    math.sin(x) * math.cos(y) * math.exp(math.fabs(1 - (((x**2 + y**2)**0.5)/math.pi)))     ))


#========================================
#              Classes
#========================================

class FitnessMax():
    def __init__(self):
        self.values = [0]

class DoubleArgument(list):
    def __init__(self, arg):
        if (isinstance(arg, (list, tuple))):
            self.dvalue = list(arg)   
        else:

            self.dvalue = [(arg>=0)]

            num = abs(int(arg))
            mantic = math.fabs(arg - int(arg))

            temp = []
            for _ in range(INT_ACCURACY):
                temp.append(num%2)
                num = num//2

            self.dvalue = self.dvalue + temp[::-1]
            for _ in range(MANTIC_ACCURACY):
                self.dvalue.append(int(mantic*2))
                mantic = mantic * 2 - int(mantic*2)

        super().__init__(self.dvalue)

    def floatNum(self):
        num = 0
        for i in range(INT_ACCURACY):
            num += self[1+i] * 2**(INT_ACCURACY-1-i)
        for i in range(MANTIC_ACCURACY):
            num += self[INT_ACCURACY+1+i] * 2**(-i-1)

        if (self[0] == 0):
            num = -num
        
        return num


class Individual():
    def __init__(self, args):
        self.fitness = FitnessMax()
        self.x = DoubleArgument(args[0])
        self.y = DoubleArgument(args[1])


#========================================
#          Additional functions
#========================================

def _oneMaxFitness(individual, border, function):
    if ((not(border[0][0] <= individual.x.floatNum() <= border[0][1])) or  (not(border[1][0] <= individual.y.floatNum() <= border[1][1]))):
        return [math.inf]
    return [function(individual.x.floatNum(), individual.y.floatNum())]

def _individualArgumentCreator():
    return ([random.randint(0, 1) for i in range(ONE_MAX_LENGTH)])

def _populationCreator(n=0):
    return list([Individual([_individualArgumentCreator(), _individualArgumentCreator()]) for i in range(n)])

def _clone(value):
    ind = Individual([value.x.floatNum(), value.y.floatNum()])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind

def _selTournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)
        offspring.append(min([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))
    return offspring

def _antiElitism(population, elitism_count):
    offspring = sorted(population, key=lambda ind: ind.fitness.values[0])
    return offspring[:elitism_count]

def _cxOnePoint(child1, child2):
    s1 = random.randint(1, len(child1.x)-2)
    s2 = random.randint(1, len(child1.y)-2)
    child1.x[s1:], child2.x[s1:] = child2.x[s1:], child1.x[s1:]
    child1.y[s2:], child2.y[s2:] = child2.y[s2:], child1.y[s2:]

def _mutFlipBit(mutant, indpb=0.01):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 0 if mutant[indx] == 1 else 1


#========================================
#          Public run function
#========================================

def run_genetic_algorithm(
    population_size=POPULATION_SIZE,
    p_crossover=P_CROSSOVER,
    p_mutation=P_MUTATION,
    max_generations=MAX_GENERATIONS,
    elitism_count=ELITISM_COUNT,
    border=None,
    function=None,
    on_progress=None,
):
    """
    Run the genetic algorithm and return statistics.

    Parameters
    ----------
    population_size : int
    p_crossover : float
    p_mutation : float
    max_generations : int
    elitism_count : int
    border : list, optional  [[xmin, xmax], [ymin, ymax]]
    function : callable, optional  f(x, y) -> float
    on_progress : callable, optional  on_progress(generation, best, mean, best_x, best_y)

    Returns
    -------
    dict with keys: best_fitness_history, mean_fitness_history,
                    best_x_history, best_y_history, best_x, best_y, best_value
    """
    if border is None:
        border = BORDER
    if function is None:
        function = FUNCTION

    population = _populationCreator(n=population_size)
    generationCounter = 0

    fitnessValues = [_oneMaxFitness(ind, border, function) for ind in population]
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    best_fitness_history = []
    mean_fitness_history = []
    best_x_history = []
    best_y_history = []

    fitnessValues = [individual.fitness.values[0] for individual in population]

    while generationCounter < max_generations:
        generationCounter += 1
        offspring = _selTournament(population, len(population) - elitism_count)
        offspring += _antiElitism(population, elitism_count)
        offspring = list(map(_clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < p_crossover:
                _cxOnePoint(child1, child2)

        for mutant in offspring:
            if random.random() < p_mutation:
                _mutFlipBit(mutant.x, indpb=1.0 / ONE_MAX_LENGTH)
                _mutFlipBit(mutant.y, indpb=1.0 / ONE_MAX_LENGTH)

        freshFitnessValues = [_oneMaxFitness(ind, border, function) for ind in offspring]
        for individual, fitnessValue in zip(offspring, freshFitnessValues):
            individual.fitness.values = fitnessValue

        population[:] = offspring
        fitnessValues = [ind.fitness.values[0] for ind in population]

        bestFitness = min(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        best_index = fitnessValues.index(bestFitness)

        best_fitness_history.append(bestFitness)
        mean_fitness_history.append(meanFitness)
        best_x_history.append(population[best_index].x.floatNum())
        best_y_history.append(population[best_index].y.floatNum())

        if on_progress is not None:
            on_progress(
                generationCounter,
                bestFitness,
                meanFitness,
                population[best_index].x.floatNum(),
                population[best_index].y.floatNum(),
            )

    best_index = fitnessValues.index(min(fitnessValues))
    return {
        "best_fitness_history": best_fitness_history,
        "mean_fitness_history": mean_fitness_history,
        "best_x_history": best_x_history,
        "best_y_history": best_y_history,
        "best_x": population[best_index].x.floatNum(),
        "best_y": population[best_index].y.floatNum(),
        "best_value": min(fitnessValues),
    }


#========================================
#                 Main
#========================================

if __name__ == '__main__':
    def _print_progress(gen, best, mean, bx, by):
        if gen % OUTPUT_FREQUENCE == 0:
            print("=" * 60)
            print(f'Поколение {gen}: Макс приспособ. = {best:.6f}, Средняя приспособ.= {mean:.6f}')
            print(f"Лучший индивидуум = {bx:.6f}, {by:.6f}")
            print("=" * 60)
            print()

    result = run_genetic_algorithm(on_progress=_print_progress)
    print(f"\nРезультат: f({result['best_x']:.6f}, {result['best_y']:.6f}) = {result['best_value']:.10f}")