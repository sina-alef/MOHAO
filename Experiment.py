from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.evaluator import MultiprocessEvaluator
from My_Evaluation import Alignment_Problem
from My_OMOPSO import My_OMOPSO
from My_NSGAII import My_NSGAII
import random
from My_Crossover import *
from My_Mutation import *
import os

from jmetal.lab.experiment import generate_boxplot
from jmetal.lab.experiment import (
     generate_median_and_wilcoxon_latex_tables,
)

from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)



max_evaluations = 6500
swarm_size = 100
problem = Alignment_Problem(16,2,False,swarm_size)
mutation_probability = 1.0 / problem.number_of_variables()
offspring_population_size = 64
A_number = 8

M_operators = [Uniform_Mutation(1.0, problem.n_D),
                      Straight_Mutation(1.0, problem.n_D),
                      Nonuniform_Mutation(1.0, problem.n_D, 64, 1.0),
                      Whole_Nonuniform_Mutation(1.0, problem.n_D, 64, 1.0)
                   ]

C_operators = [Simple_Crossover(problem.n_D),
                       Two_Point_Crossover(problem.n_D),
                       Arithmetic_crossover(problem.n_D),
                       Heuristic_crossover(problem.n_D)]

def configure_experiment(problems: dict, n_run: int):
#    jobs = []
    max_evaluations = 6500

    for run in range(n_run):
        for problem_tag, problem in problems.items():
            if run not in []:
                yield Job(
                        algorithm=NSGAII(
                            problem=problem,
                            population_size=100,
                            offspring_population_size=64,
                            mutation=PolynomialMutation(
                                probability=mutation_probability, distribution_index=20
                            ),
                            crossover=SBXCrossover(probability=1.0, distribution_index=20),
                            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
                            population_evaluator=MultiprocessEvaluator(8),
                        ),
                        algorithm_tag="NSGAII",
                        problem_tag=problem_tag,
                        run=run,
                    )
            if run not in []:
                yield Job(
                        algorithm=My_OMOPSO(problem=problem,
                                            swarm_size=swarm_size,
                                            epsilon=0.0075,
                                            mutation_operators=M_operators,
                                            crossover_operators=C_operators,
                                            leaders=CrowdingDistanceArchive(100),
                                            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
                                            applying_number=A_number,
                            swarm_evaluator=MultiprocessEvaluator(8)
                        ),
                        algorithm_tag='My_OMOPSO',
                        problem_tag=problem_tag,
                        run=run,
                    )
            if run not in []:
                yield Job(
                        algorithm=OMOPSO(problem=problem,
                                            swarm_size=swarm_size,
                                            epsilon=0.0075,
                                            uniform_mutation=UniformMutation(probability=mutation_probability, perturbation=2500),
                                            non_uniform_mutation=NonUniformMutation(mutation_probability,2500,max_evaluations//swarm_size),
                                            leaders=CrowdingDistanceArchive(100),
                                            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
                            swarm_evaluator=MultiprocessEvaluator(8)
                        ),
                        algorithm_tag='OMOPSO',
                        problem_tag=problem_tag,
                        run=run,
                    )
            if run not in []:
                yield Job(
                        algorithm=My_NSGAII(problem=problem,
                                            population_size=swarm_size,
                                            offspring_population_size = 64,
                                            mutation=PolynomialMutation(probability=1.0 * 1.0 / problem.number_of_variables(), distribution_index=20),
                                            crossover=SBXCrossover(probability=1.0, distribution_index=20),
                                            mutation_operators = M_operators,
                                            crossover_operators = C_operators,
                                            applying_number = A_number,
                                            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
                            population_evaluator=MultiprocessEvaluator(8)
                        ),
                        algorithm_tag='My_NSGAII',
                        problem_tag=problem_tag,
                        run=run,
                    )

def get_combined_front(
        input_dir: str, output_dir: str):
    
    all_solutions = []

    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            try:
                # Linux filesystem
                algorithm, problem = dirname.split("/")[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem = dirname.split("\\")[-2:]

            if "FUN" in filename:
                solutions = read_solutions(os.path.join(dirname, filename))
                print('len_solutions:', len(solutions))
                all_solutions.extend(solutions)
                print('len_All_solutions:', len(all_solutions))

    front = get_non_dominated_solutions(all_solutions)
    print('len_front:', len(front))
    print_function_values_to_file(front, output_dir)
    
def get_reference_point(
    input_dir: str, output_dir: str):

    max_first = float('-inf')
    max_second = float('-inf')
    
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            try:
                # Linux filesystem
                algorithm, problem = dirname.split("/")[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem = dirname.split("\\")[-2:]

            if "FUN" in filename:
                with open(os.path.join(dirname, filename), 'r') as file:
                    for line in file:
                        numbers = line.split()
                        first_number = float(numbers[0])
                        second_number = float(numbers[1])
                        max_first = max(max_first, first_number)
                        max_second = max(max_second, second_number)

    with open(output_dir, mode = 'w') as file:

        file.write(f"{max_first} {max_second}\n")

    return [max_first, max_second]
                
if __name__ == '__main__':
    
    jobs = configure_experiment(problems={'Alignment_Problem': problem}, n_run=4)
    output_directory = "data"

    for job in jobs:
        path = os.path.join(output_directory, job.algorithm_tag, job.problem_tag)
        job.execute(path)


    get_combined_front(output_directory, r"resources\reference_front\Alignment_Problem.pf")
    RP = get_reference_point(output_directory, r"resources\reference_point\Alignment_Problem.rp")
    
    # Generate summary file
    generate_summary_from_experiment(
        input_dir=output_directory,
        reference_fronts=r"resources\reference_front",
        quality_indicators=[GenerationalDistance(), EpsilonIndicator(), HyperVolume(RP)],
    )
    generate_boxplot(filename="QualityIndicatorSummary.csv")
    generate_median_and_wilcoxon_latex_tables(filename="QualityIndicatorSummary.csv")
    
