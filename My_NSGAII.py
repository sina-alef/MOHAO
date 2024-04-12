from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.problem import Problem
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from typing import Generator, List, TypeVar
from jmetal.core.operator import Crossover, Mutation
from jmetal.config import store
from jmetal.util.comparator import DominanceComparator
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.evaluator import Evaluator
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator import PolynomialMutation, SBXCrossover
import random
from My_Crossover import *
from My_Mutation import *
from My_Evaluation import Alignment_Problem

S = TypeVar("S")
R = TypeVar("R")

class My_NSGAII(NSGAII):
    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation_operators: List[Mutation],
                 crossover_operators: List[Crossover],
                 applying_number: int,
                 mutation,
                 crossover,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: DominanceComparator = DominanceComparator(),
                 ):
        
        self.mating_pool_size = population_size
        super(My_NSGAII, self).__init__(problem = problem,
                                        population_size = population_size,
                                        offspring_population_size = offspring_population_size,
                                        mutation = mutation,
                                        crossover = crossover,
                                        population_evaluator=population_evaluator,
                                        population_generator=population_generator,
                                        termination_criterion=termination_criterion,
                                        dominance_comparator=dominance_comparator,
#                                        mating_pool_size = population_size
                                        )
        self.mating_pool_size = population_size
        self.mutation_operators = mutation_operators
        self.crossover_operators = crossover_operators
        self.applying_number = applying_number
        self.current_iteration = 0
#        self.mating_pool_size = population_size

#    def reproduction(self, mating_population: List[S]) -> List[S]:
    
    def reproduction(self, mating_population: List[S]):

        self.current_iteration += 1
        print("iteration:", self.current_iteration)
        offspring_population = []
        
        for num in range(self.applying_number):
            for op in self.crossover_operators:
                if hasattr(op, 'set_current_iteration'):
                    op.set_current_iteration = self.current_iteration
                parents = random.sample(mating_population, 2)
                offspring_population.extend(op.execute(parents))
            
        for num in range(self.applying_number):
            for op in self.mutation_operators:
                if hasattr(op, 'set_current_iteration'):
                    op.set_current_iteration = self.current_iteration
                offspring = random.choice(mating_population)
                offspring_population.append(op.execute(offspring))

        return offspring_population
    
                
if __name__ == '__main__':
    
    
    problem = Alignment_Problem(16,2,False)
    population_size = 100
    offspring_population_size = 60

    M_operators = [Uniform_Mutation(1.0, problem.n_D),
                   Straight_Mutation(1.0, problem.n_D),
                   Nonuniform_Mutation(1.0, problem.n_D, 100, 1.0),
                   Whole_Nonuniform_Mutation(1.0, problem.n_D, 100, 1.0)]

    C_operators = [Simple_Crossover(problem.n_D),
                   Two_Point_Crossover(problem.n_D),
                   Arithmetic_crossover(problem.n_D),
                   Heuristic_crossover(problem.n_D)]

    A_number = 5

    max_evaluations = 6100

    algorithm = My_NSGAII(problem=problem,
                          population_size=population_size,
                          offspring_population_size = offspring_population_size,
                          mutation=PolynomialMutation(probability=1.0 * 1.0 / problem.number_of_variables(), distribution_index=20),
                          crossover=SBXCrossover(probability=1.0, distribution_index=20),                          
                          mutation_operators = M_operators,
                          crossover_operators = C_operators,
                          applying_number = A_number,
                          termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
                          population_evaluator=MultiprocessEvaluator(8)
                          )
    
    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
