from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.util.termination_criterion import StoppingByEvaluations
from typing import List, Optional, TypeVar
from jmetal.core.solution import FloatSolution
from My_Evaluation import Alignment_Problem
from My_Crossover import *
from My_Mutation import *
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.config import store
from jmetal.util.evaluator import Evaluator
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)

class My_OMOPSO(OMOPSO):
    def __init__(self,
                 problem,
                 swarm_size,
                 epsilon,
                 mutation_operators,
                 crossover_operators,
                 leaders,
                 termination_criterion,
                 applying_number,
                 swarm_evaluator: Evaluator = store.default_evaluator,
                 uniform_mutation = None,
                 non_uniform_mutation = None
                 
                 ):

        super(My_OMOPSO, self).__init__(problem = problem,
                                  swarm_size=swarm_size,
                                  epsilon = epsilon,
                                  uniform_mutation = uniform_mutation,
                                  non_uniform_mutation = non_uniform_mutation,
                                  leaders = leaders,
                                  termination_criterion = termination_criterion,
                                        swarm_evaluator = swarm_evaluator
                                  )
        self.mutation_operators = mutation_operators
        self.crossover_operators = crossover_operators
        self.applying_number = applying_number
        self.current_iteration = 0

    def perturbation(self, swarm: List[FloatSolution]) -> None:

        self.current_iteration += 1
        print("iteration:", self.current_iteration)
        index = self.swarm_size -1
            
        for num in range(self.applying_number):
            for op in self.mutation_operators:
                if hasattr(op, 'set_current_iteration'):
                    op.set_current_iteration = self.current_iteration
                offspring = swarm[index]
                swarm[index] = op.execute(solution = offspring)
                index -= 1
                
        for num in range(self.applying_number):
            for op in self.crossover_operators:
                if hasattr(op, 'set_current_iteration'):
                    op.set_current_iteration = self.current_iteration
                parents = [swarm[index], swarm[index -1]]
                [swarm[index], swarm[index -1]] = op.execute(parents = parents)
                index -= 2


if __name__ == '__main__':
    
    
    max_evaluations = 400
    swarm_size = 100
    problem = Alignment_Problem(16,2,False)
    M_operators = [
        Uniform_Mutation(1.0, problem.n_D),
                      Straight_Mutation(1.0, problem.n_D),
                      Nonuniform_Mutation(1.0, problem.n_D, 100, 1.0),
                      Whole_Nonuniform_Mutation(1.0, problem.n_D, 100, 1.0)
                   ]

    C_operators = [Simple_Crossover(problem.n_D),
                       Two_Point_Crossover(problem.n_D),
                       Arithmetic_crossover(problem.n_D),
                       Heuristic_crossover(problem.n_D)]


    algorithm=My_OMOPSO(problem=problem,
                    swarm_size=swarm_size,
                    epsilon =1.0,
                    mutation_operators=M_operators,
                    crossover_operators=C_operators,
                    leaders=CrowdingDistanceArchive(100),
                    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
                    applying_number=5,
#                    swarm_evaluator=MultiprocessEvaluator(8)
    )
    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
