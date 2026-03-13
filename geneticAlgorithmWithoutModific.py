import random
import math
import matplotlib.pyplot as plt



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
# FUNCTION = lambda x,y: (x-5.5)**4 + (y - 6.25)**2
# FUNCTION = lambda x, y: math.sin(x+y) + (x-y)**2 - 1.5*x + 2.5*y + 1 
# FUNCTION = lambda x,y: (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y) /2


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

def oneMaxFitness(individual):
    if ((not(BORDER[0][0] <= individual.x.floatNum() <= BORDER[0][1])) or  (not(BORDER[1][0] <= individual.y.floatNum() <= BORDER[1][1]))):
        return [math.inf]
    return [FUNCTION(individual.x.floatNum(), individual.y.floatNum())]

def individualArgumentCreator():
    return ([random.randint(0, 1) for i in range(ONE_MAX_LENGTH)])
    # return ([1]+[0 for i in range(ONE_MAX_LENGTH)])

def populationCreator(n = 0):
    return list([Individual([individualArgumentCreator(), individualArgumentCreator()]) for i in range(n)])

def clone (value):
    # print(value.x.floatNum(), value.y.floatNum())
    ind = Individual([value.x.floatNum(), value.y.floatNum()])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind

def selTournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

        offspring.append(min([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

    return offspring

def antiElitism(population, n=0):
    offspring = sorted(population, key=lambda ind: ind.fitness.values[0])
    return offspring[:ELITISM_COUNT]
    

def cxOnePoint(child1, child2):
    s1 = random.randint(1, len(child1.x)-2)
    s2 = random.randint(1, len(child1.y)-2)
    child1.x[s1:], child2.x[s1:] = child2.x[s1:], child1.x[s1:]
    child1.y[s2:], child2.y[s2:] = child2.y[s2:], child1.y[s2:]

def mutFlipBit(mutant, indpb=0.01):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 0 if mutant[indx] == 1 else 1

# def lookBorder(population, border):
#     for ind in population:
#         if (border[0][0] < ind.x.floatNum()):
#             ind.x = DoubleArgument(border[0][0])
#         elif (border[0][1] > ind.x.floatNum()):
#             ind.x = DoubleArgument(border[0][1])

#         if (border[1][0] < ind.y.floatNum()):
#             ind.y = DoubleArgument(border[1][0])
#         elif (border[1][1] > ind.y.floatNum()):
#             ind.y = DoubleArgument(border[1][1])

    

#========================================
#                 Main
#========================================

population = populationCreator(n=POPULATION_SIZE)
# lookBorder(population, BORDER)
generationCounter = 0

fitnessValues = list(map(oneMaxFitness, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

maxFitnessValues = []
meanFitnessValues = []
maxFitnessCoordinatesX = []
maxFitnessCoordinatesY = []

fitnessValues = [individual.fitness.values[0] for individual in population]


# max(fitnessValues) < ONE_MAX_LENGTH and 
while generationCounter < MAX_GENERATIONS:
    generationCounter += 1
    offspring = selTournament(population, len(population)-ELITISM_COUNT)
    offspring += antiElitism(population, ELITISM_COUNT)
    offspring = list(map(clone, offspring))

    for child1, child2 in zip (offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            cxOnePoint(child1, child2)

    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutFlipBit(mutant.x, indpb = 1.0/ONE_MAX_LENGTH)

            mutFlipBit(mutant.y, indpb = 1.0/ONE_MAX_LENGTH)

    # lookBorder(offspring, BORDER)

    freshFitnessValues = list(map(oneMaxFitness, offspring))
    for individual,fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue

    population[:] = offspring
    # print([[ind.x.floatNum(), ind.y.floatNum()] for ind in population])
    fitnessValues = [ind.fitness.values[0] for ind in population]

    maxFitness = min(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)

    best_index = fitnessValues.index(min(fitnessValues))
    maxFitnessCoordinatesX.append(population[best_index].x.floatNum())
    maxFitnessCoordinatesY.append(population[best_index].y.floatNum())

    if (generationCounter % OUTPUT_FREQUENCE == 0):
        print("=" * 60)
        print(f'Поколение {generationCounter}: Макс приспособ. = {maxFitness}, Средняя приспособ.= {meanFitness}')
        print("Лучший индивидуум = ", population[best_index].x.floatNum(), ', ', population[best_index].y.floatNum())
        print("=" * 60)
        print('\n')



#========================================
#                Graph
#========================================


plt.scatter(maxFitnessCoordinatesX, maxFitnessCoordinatesY, color='red', label='Точки')

# Настройка (необязательно)
plt.title('График с N точками')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.grid(True)
plt.legend()

# Показать график
plt.show()