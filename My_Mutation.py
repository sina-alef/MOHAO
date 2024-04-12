import random
import copy

from jmetal.core.operator import Mutation
from jmetal.core.solution import (

    FloatSolution,
    Solution,
)
from jmetal.util.ckecking import Check
    
class Uniform_Mutation(Mutation[FloatSolution]):
    def __init__(self, probability: float, n_D:int):
        super(Uniform_Mutation, self).__init__(probability=probability)
        self.n_D = n_D
        
    def curve_elimination(self, solution: FloatSolution, m: int, n: int) -> FloatSolution:

        if m > n:
            m, n = n, m
            
        for ip in range(m, n):
            for i in range(self.n_D):
                solution.variables[self.n_D*ip+i] = (
                    solution.variables[self.n_D*m+i] +
                (solution.variables[self.n_D*n+i] - solution.variables[self.n_D*m+i])*(ip - m)/(n - m)
                    )

        return solution

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(type(solution) is FloatSolution, "Solution type invalid")
        
        offspring = copy.deepcopy(solution)
        k = random.randint(0,len(offspring.variables)//self.n_D -1)

        for i in range(self.n_D):
            offspring.variables[self.n_D*k+i] = random.uniform(offspring.lower_bound[self.n_D*k+i], offspring.upper_bound[self.n_D*k+i])

        m, n = random.randint(0, k), random.randint(k, len(offspring.variables)//self.n_D -1)
        
        offspring = self.curve_elimination(offspring, m, k)
        offspring = self.curve_elimination(offspring, k, n)


##        for i in range(0,len(offspring.variables)-2,2):
##            if offspring.variables[i] == offspring.variables[i+2] and offspring.variables[i+1] == offspring.variables[i+3]:
##                print(self.get_name())
##                print(offspring.variables)
##
##
##        for i in offspring.variables:
##            if i in [-9176837.0,-9170875.0]+[3294621.7,3301272.0]:
##                print(self.get_name())
##                print (offspring.variables)
                
        return offspring

    def get_name(self):
        return "Uniform_Mutation"

class Straight_Mutation(Mutation[FloatSolution]):
    def __init__(self, probability: float, n_D:int):
        super(Straight_Mutation, self).__init__(probability=probability)
        self.n_D = n_D
        
    def curve_elimination(self, solution: FloatSolution, m: int, n: int) -> FloatSolution:

#        m, n = random.randint(0, k), random.randint(k, len(solution.variables)//self.n_D)
        if m > n:
            m, n = n, m
            
        for ip in range(m, n):
            for i in range(self.n_D):
                solution.variables[self.n_D*ip+i] = (
                    solution.variables[self.n_D*m+i] +
                (solution.variables[self.n_D*n+i] - solution.variables[self.n_D*m+i])*(ip - m)/(n - m)
                    )

        return solution

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(type(solution) is FloatSolution, "Solution type invalid")
        
        offspring = copy.deepcopy(solution)
        m, n = 0,0

        while m == n:
            m, n = random.randint(0, len(offspring.variables)//self.n_D -1), random.randint(0, len(offspring.variables)//self.n_D -1)
        
        offspring = self.curve_elimination(offspring, m, n)

##        for i in range(0,len(offspring.variables)-2,2):
##            if offspring.variables[i] == offspring.variables[i+2] and offspring.variables[i+1] == offspring.variables[i+3]:
##                print(self.get_name())
##                print(offspring.variables)
##
##
##        for i in offspring.variables:
##            if i in [-9176837.0,-9170875.0]+[3294621.7,3301272.0]:
##                print(self.get_name())
##                print (offspring.variables)
                
        return offspring

    def get_name(self):
        return "Straight_Mutation"


class Nonuniform_Mutation(Mutation[FloatSolution]):
    def __init__(self, probability: float, n_D:int, max_iterations: int, degreeofnonuniformity: float):
        super(Nonuniform_Mutation, self).__init__(probability=probability)
        self.n_D = n_D
        self.max_iterations = max_iterations
        self.degreeofnonuniformity = degreeofnonuniformity
        self.current_iteration = 0

    def curve_elimination(self, solution: FloatSolution, m: int, n: int) -> FloatSolution:

#        m, n = random.randint(0, k), random.randint(k, len(solution.variables)//self.n_D)
        if m > n:
            m, n = n, m
            
        for ip in range(m, n):
            for i in range(self.n_D):
                solution.variables[self.n_D*ip+i] = (
                    solution.variables[self.n_D*m+i] +
                (solution.variables[self.n_D*n+i] - solution.variables[self.n_D*m+i])*(ip - m)/(n - m)
                    )

        return solution
        
    def delta (self, dist, degreeofnonuniformity):
        return dist*random.random()*(1.0 - self.current_iteration / self.max_iterations)**degreeofnonuniformity

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(type(solution) is FloatSolution, "Solution type invalid")

        offspring = copy.deepcopy(solution)
        k = random.randint(0,len(offspring.variables)//self.n_D -1)
        
        for i in range(self.n_D):
            if random.random() <= 0.5:
                offspring.variables[self.n_D*k + i] += self.delta(offspring.variables[self.n_D*k + i] - offspring.lower_bound[self.n_D*k + i], self.degreeofnonuniformity)
            else:
                offspring.variables[self.n_D*k + i] -= self.delta(offspring.variables[self.n_D*k + i] - offspring.upper_bound[self.n_D*k + i], self.degreeofnonuniformity)

        m, n = random.randint(0, k), random.randint(k, len(offspring.variables)//self.n_D -1)
        
        offspring = self.curve_elimination(offspring, m, k)
        offspring = self.curve_elimination(offspring, k, n)

##        for i in range(0,len(offspring.variables)-2,2):
##            if offspring.variables[i] == offspring.variables[i+2] and offspring.variables[i+1] == offspring.variables[i+3]:
##                print(self.get_name())
##                print(offspring.variables)
##
##
##        for i in offspring.variables:
##            if i in [-9176837.0,-9170875.0]+[3294621.7,3301272.0]:
##                print(self.get_name())
##                print (offspring.variables)
                
        return offspring

    def set_current_iteration(self, current_iteration: int):
        self.current_iteration = current_iteration
        
    def get_name(self):
        return "Nonuniform_Mutation"


class Whole_Nonuniform_Mutation(Mutation[FloatSolution]):
    def __init__(self, probability: float, n_D:int, max_iterations: int, degreeofnonuniformity: float):
        super(Whole_Nonuniform_Mutation, self).__init__(probability=probability)
        self.n_D = n_D
        self.max_iterations = max_iterations
        self.degreeofnonuniformity = degreeofnonuniformity
        self.current_iteration = 0

    def curve_elimination(self, solution: FloatSolution, m: int, n: int) -> FloatSolution:

#        m, n = random.randint(0, k), random.randint(k, len(solution.variables)//self.n_D)
        if m > n:
            m, n = n, m
            
        for ip in range(m, n):
            for i in range(self.n_D):
                solution.variables[self.n_D*ip+i] = (
                    solution.variables[self.n_D*m+i] +
                (solution.variables[self.n_D*n+i] - solution.variables[self.n_D*m+i])*(ip - m)/(n - m)
                    )

        return solution
        
    def delta (self, dist, degreeofnonuniformity):
        return dist*random.random()*(1.0 - self.current_iteration / self.max_iterations)**degreeofnonuniformity

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(type(solution) is FloatSolution, "Solution type invalid")

        offspring = copy.deepcopy(solution)
        
##        for i in offspring.variables:
##            if i in [-9176837.0,-9170875.0]+[3294621.7,3301272.0]:
##                print("UPPER", self.get_name())
##                print ("UPPER",offspring.variables)
                
        for k in range(len(offspring.variables)//self.n_D -1):
            for i in range(self.n_D):
                if random.random() <= 0.5:
                    offspring.variables[self.n_D*k + i] += self.delta(offspring.variables[self.n_D*k + i] - offspring.lower_bound[self.n_D*k + i], self.degreeofnonuniformity)
                else:
                    offspring.variables[self.n_D*k + i] -= self.delta(offspring.variables[self.n_D*k + i] - offspring.upper_bound[self.n_D*k + i], self.degreeofnonuniformity)

##        for i in range(0,len(offspring.variables)-2,2):
##            if offspring.variables[i] == offspring.variables[i+2] and offspring.variables[i+1] == offspring.variables[i+3]:
##                print(self.get_name())
##                print(offspring.variables)
##
##        for i in offspring.variables:
##            if i in [-9176837.0,-9170875.0]+[3294621.7,3301272.0]:
##                print(self.get_name())
##                print (offspring.variables)
                
        return offspring
    
    def set_current_iteration(self, current_iteration: int):
        self.current_iteration = current_iteration
        
    def get_name(self):
        return "Whole_Nonuniform_Mutation"
    
