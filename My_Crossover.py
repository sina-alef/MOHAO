import copy
import random
from typing import List

from jmetal.core.operator import Crossover
from jmetal.core.solution import (
    BinarySolution,
    CompositeSolution,
    FloatSolution,
    IntegerSolution,
    PermutationSolution,
    Solution,
)
from jmetal.util.ckecking import Check

"""
.. module:: crossover
   :platform: Unix, Windows
   :synopsis: Module implementing crossover operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""

class Simple_Crossover(Crossover[FloatSolution, FloatSolution]):
    def __init__(self, n_D):
        super(Simple_Crossover, self).__init__(probability=1.0)
        self.n_D = n_D

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid: " + str(type(parents[0])))
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        offspring = copy.deepcopy(parents)

        parent1, parent2 = offspring

        k = random.randint(1,len(parent1.variables)//self.n_D -2)

        parent1.variables[:k], parent2.variables[:k] = parent2.variables[:k], parent1.variables[:k]

##        for child in offspring:
##            for i in range(0,len(child.variables)-2,2):
##                if child.variables[i] == child.variables[i+2] and child.variables[i+1] == child.variables[i+3]:
##                    print(self.get_name())
##                    print(child.variables)
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Simple_Crossover"

    
class Two_Point_Crossover(Crossover[FloatSolution, FloatSolution]):
    def __init__(self,n_D):
        super(Two_Point_Crossover, self).__init__(probability=1.0)
        self.n_D = n_D
    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid: " + str(type(parents[0])))
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        offspring = copy.deepcopy(parents)
        parent1, parent2 = offspring

##        for child in offspring:
##            for i in range(0,len(child.variables)-2,2):
##                if child.variables[i] == child.variables[i+2] and child.variables[i+1] == child.variables[i+3]:
##                    print(self.get_name(),'UUUOOOO')
##                    print(child.variables)
        
        m, n = 0, 0

        while m == n:
            m, n = random.randint(1,len(parent1.variables)//self.n_D -2), random.randint(1,len(parent1.variables)//self.n_D -2)

        if m > n:
            m, n = n, m
        
        parent1.variables[m:n], parent2.variables[m:n] = parent2.variables[m:n], parent1.variables[m:n]

##        for child in offspring:
##            for i in range(0,len(child.variables)-2,2):
##                if child.variables[i] == child.variables[i+2] and child.variables[i+1] == child.variables[i+3]:
##                    print(self.get_name(),'DOOOWN')
##                    print(child.variables)
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Two_Point_Crossover"
    
class Arithmetic_crossover(Crossover[FloatSolution, FloatSolution]):
    def __init__(self,n_D):
        super(Arithmetic_crossover, self).__init__(probability=1.0)
        self.n_D = n_D
        
    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid: " + str(type(parents[0])))
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        parent1, parent2 = parents
        offspring = copy.deepcopy(parents)
        child1, child2 = offspring

        omega= random.random()

        for i in range(len(child1.variables)):
            child1.variables[i] = omega * parent1.variables[i] + (1-omega) * parent2.variables[i]
            child2.variables[i] = omega * parent2.variables[i] + (1-omega) * parent1.variables[i]

##        for child in offspring:
##            for i in range(0,len(child.variables)-2,2):
##                if child.variables[i] == child.variables[i+2] and child.variables[i+1] == child.variables[i+3]:
##                    print(self.get_name())
##                    print(child.variables)            
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Arithmetic_crossover"

class Heuristic_crossover(Crossover[FloatSolution, FloatSolution]):
    def __init__(self, n_D):
        super(Heuristic_crossover, self).__init__(probability=1.0)
        self.n_D = n_D
        
    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid: " + str(type(parents[0])))
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        
        offspring = copy.deepcopy(parents)
#        child1, child2 = offspring

        number_of_objectives = len(parents[0].objectives)

        for ob in range(number_of_objectives):

            child = offspring[ob]
            omega= random.random()
            parents.sort(key = lambda ch : ch.objectives[ob])
            parent1, parent2 = parents
            
            for i in range(len(child.variables)):
                child.variables[i] = omega*(parent1.variables[i] - parent2.variables[i]) + parent1.variables[i]

            if any(child.variables[i] < child.lower_bound[i] or child.variables[i] > child.upper_bound[i] for i in range(len(child.variables))):
                child = parent1

##        for child in offspring:
##            for i in range(0,len(child.variables)-2,2):
##                if child.variables[i] == child.variables[i+2] and child.variables[i+1] == child.variables[i+3]:
##                    print(self.get_name())
##                    print(child.variables)            
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Heuristic_crossover"
