# Genetic Algorithm for Optimal Pairing

## Overview

This project utilizes a genetic algorithm (GA) to solve an optimization problem involving the pairing of individuals based on preference scores. It simulates the process of natural selection where the fittest individuals are selected for reproduction to produce offspring of the next generation. The goal is to maximize the overall compatibility of pairings in a population.

## Project Description

### Objective

To develop a genetic algorithm that efficiently finds an optimal or near-optimal solution to pairing individuals based on given preference matrices. The algorithm evolves a population of potential solutions over several generations to maximize a fitness function that reflects pairing compatibility.

### Features

- **Dynamic Population Management**: Handles a population where each individual represents a potential solution encoded as a permutation of pairings.
- **Fitness Evaluation**: Each solution's fitness is calculated based on the sum of preference scores, normalized by a theoretical maximum score.
- **Selection and Reproduction**: Implements elitism and crossover operations to generate the next generation of solutions.
- **Mutation**: Introduces random changes to offspring to explore new genetic combinations and avoid local minima.
- **Convergence Monitoring**: Tracks fitness over generations and adjusts parameters if the population's fitness stagnates.

### Implementation Details

#### Main Components

- `generate_population`: Generates an initial random population of potential solutions.
- `calculate_fitness`: Computes the fitness of each individual based on the preference scores from both parties.
- `select_elite`: Selects the top-performing individuals to preserve the best genes.
- `crossover`: Combines two parents to create offspring using Partially Mapped Crossover (PMX), ensuring valid permutations.
- `mutate`: Applies random mutations to individuals to maintain genetic diversity.
- `new_generation`: Produces a new generation of the population, incorporating elitism, crossover, and mutation.

#### Utilities

- `validate_population`: Ensures all individuals in the population represent valid solutions.
- `format_best_individual`: Formats the best solution for display.
- `show_results`: Uses Tkinter to display the results in a simple GUI.

### Usage

1. **Prepare Input Data**:
   Ensure the `GA_input.txt` file is formatted correctly with preference data for all individuals involved.

2. **Configuration**:
   Adjust parameters such as population size, mutation rate, and number of generations in the script to fit the problem size and desired accuracy.

3. **Run the Algorithm**:
   Execute the script to start the genetic algorithm process:
   ```bash
   python ga_optimal_pairing.py
View Results: After the computation, the best pairing and its fitness score will be displayed both on the command line and in a graphical window.
Visualization
The script generates plots showing the evolution of fitness across generations, helping visualize the algorithm's performance and convergence.
Dependencies
Python: Version 3.x+
Matplotlib: For generating plots.
Tkinter: For creating GUIs.
Numpy: Used for numerical operations.
Testing the Library
Modify input data in GA_input.txt to test different scenarios.
Observe how changes in parameters affect the performance and results of the algorithm.
Contributions
Contributions to improve the algorithm, add features, or enhance the visualization are welcome. Please fork the repository and submit a pull request with your changes.

License
This project is released under the MIT License.

Contact
For more information or to raise issues, please contact [RanWurembrand@gmail.com].
