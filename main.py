import random
import matplotlib.pyplot as plt
import tkinter as tk

# Configuration parameters
population_size = 30 # number of individuals
elite_fraction = 0.1  # Keep top 25% of the population initially
mutation_rate = 0.15 # 10% of the new population undergoes mutation
num_generations = 900  # Number of generations to evolve
highest_matching_score = 1800  #30 couples each couple can get max 60 points


# checks if an individual (a solution) is valid by making sure it
# contains exactly 30 unique values from 0 to 29, ensuring it represents a valid permutation.
def is_valid_individual(individual):
    if sorted(individual) != list(range(30)) or len(individual) != 30:
        missing = set(range(30)) - set(individual)
        extra = set(individual) - set(range(30))
        duplicates = {x for x in individual if individual.count(x) > 1}
        if missing or extra or duplicates:
            return False, f"Errors in individual. Missing: {missing}, Extra: {extra}, Duplicates: {duplicates}"
    return True, "Valid individual."

# checks if there is an invalid individual in the population
def validate_population(population):
    for individual in population:
        valid, message = is_valid_individual(individual)
        if not valid:
            return False, message
    return True, "All individuals are valid."

# Function to read preferences from file
def read_preferences(file_name):
    preferences = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            preferences.append(list(map(int, line.strip().split())))
    return preferences

# read preference data for 30 males and 30 females from a text file,
# assuming each line corresponds to the preferences of one individual.
def get_data_from_textfile():
    input_file = "GA_input.txt"
    male_preferences, female_preferences = [], []
    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)
        num_couples = num_lines // 2
        male_preferences = read_preferences(input_file)[
                           :num_couples]
        female_preferences = read_preferences(input_file)[
                             num_couples:]
    return male_preferences, female_preferences, num_couples

# Generates an initial population of random permutations of the numbers 0 to 29,
# each permutation representing an individual.
def generate_population(population_size):
    return [random.sample(range(30), 30) for _ in range(population_size)]

# Calculates the fitness of a solution based on the sum of preferences from male_data and female_data,
# normalized by the highest possible score.
def calculate_fitness(solution, male_data, female_data):
    return 3 +(100 * sum(male_data[man][woman] + female_data[woman][man] for man, woman in enumerate(solution))) / highest_matching_score

#Selects the top-performing individuals from the population to form the elite group.
#to maintain the monotonicly of the best solution

def select_elite(population, fitness_scores, elite_size):
    paired_population = list(zip(population, fitness_scores))
    sorted_population = sorted(paired_population, key=lambda x: x[1], reverse=True)
    return [individual for individual, score in sorted_population[:elite_size]]

# Partially Mapped Crossover (PMX) function to combine two parents into a new child,
# ensuring that all genetic material is properly mapped to avoid duplicates.
def pmx(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[point1:point2 + 1] = parent1[point1:point2 + 1]
    mapping = {parent1[i]: parent2[i] for i in range(point1, point2 + 1)}
    for i in range(size):
        if child[i] is None:
            value = parent2[i]
            while value in child:
                value = mapping.get(value, value)
            child[i] = value
    return child


# for 2 parents it returns 2 childrens
def crossover(parent1, parent2):
    return pmx(parent1, parent2), pmx(parent2, parent1)


# Randomly swaps two elements in an individual, altering its genetic sequence.
# This is done with a probability defined by the mutation rate.
def mutate(individual, mutation_rate):
    size = len(individual)
    for _ in range(int(size * mutation_rate)):
        index1, index2 = random.sample(range(size), 2)
        individual[index1], individual[index2] = individual[index2], individual[index1]


def new_generation(population, fitness_scores, elite_fraction):
    # Determine the number of elites and calculate the indices for elite and non-elite groups
    num_elites = int(len(population) * elite_fraction)
    num_non_elites = len(population) - num_elites - 2  # Excluding the top 2 solutions

    # Sort the population based on fitness scores, keeping the best 2 and the rest
    paired_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    top_two_solutions = [individual for individual, _ in paired_population[:2]]
    remaining_population = [individual for individual, _ in paired_population[2:]]

    # Divide the remaining population into elite and non-elite groups
    elite_group = remaining_population[:num_elites]
    non_elite_group = remaining_population[num_elites:]
    random.shuffle(non_elite_group)

    # Perform crossover within each group
    children = []
    # Crossover for elites
    for i in range(0, len(elite_group) - 1, 2):
        child1, child2 = crossover(elite_group[i], elite_group[i + 1])
        children.extend([child1, child2])
    # Crossover for non-elites
    for i in range(0, len(non_elite_group) - 1, 2):
        child1, child2 = crossover(non_elite_group[i], non_elite_group[i + 1])
        children.extend([child1, child2])

    # Mutation for the generated children
    for child in children:
        mutate(child, mutation_rate)

    # Combine the best two solutions with the new children
    new_population = top_two_solutions + children

    # Calculate fitness for the new population
    new_fitness_scores = [calculate_fitness(ind, male_array, female_array) for ind in new_population]

    return new_population, new_fitness_scores


male_array, female_array, population_size = get_data_from_textfile()
population = generate_population(population_size)
fitness_scores = [calculate_fitness(ind, male_array, female_array) for ind in population]

best_fitness = 0
best_stagnant_generations = 0
fitness_history, min_fitness_history, avg_fitness_history = [], [], []

# Iterates through the number of generations,
# updates populations, checks for stagnation, and possibly adjusts the elite fraction.
for generation in range(num_generations):
    if best_fitness == max(fitness_scores):
        best_stagnant_generations += 1
    else:
        best_fitness = max(fitness_scores)
        best_stagnant_generations = 0

    # Adjust elite fraction and mutation rate if there is stagnation for 25% of the total generations
    if best_stagnant_generations >= num_generations * 0.25:
        elite_fraction = min(elite_fraction + 0.1, 1.0)  # Ensure that the fraction doesn't exceed 100%
        mutation_rate = min(mutation_rate + 0.1, 1.0)    # Ensure that the mutation rate doesn't exceed 100%
        best_stagnant_generations = 0  # Reset the counter after adjusting parameters

        # Remove the lowest 2 solutions and add 2 new random solutions
        sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        population = [individual for individual, _ in sorted_population[:-2]]  # Remove the two worst
        population.extend(generate_population(2))  # Add two new random individuals

        # Recalculate fitness scores for the new population
        fitness_scores = [calculate_fitness(ind, male_array, female_array) for ind in population]

    # Evolve the population
    population, fitness_scores = new_generation(population, fitness_scores, elite_fraction)
    max_fitness, min_fitness, avg_fitness = max(fitness_scores), min(fitness_scores), sum(fitness_scores) / len(fitness_scores)
    fitness_history.append(max_fitness)
    min_fitness_history.append(min_fitness)
    avg_fitness_history.append(avg_fitness)

    valid, error_message = validate_population(population)
    if not valid:
        print(f"Invalid population at Generation {generation + 1}: {error_message}")
        break

    print(f"Generation {generation + 1}: Max Fitness = {max_fitness}, Elite Fraction = {elite_fraction}, Mutation Rate = {mutation_rate}")



best_index = fitness_scores.index(max(fitness_scores))
best_individual = population[best_index]
best_score = fitness_scores[best_index]

print("Best Solution:", best_individual)
print("Best Fitness Score:", best_score)


def format_best_individual(individual):
    return ', '.join(f"({i}, {val})" for i, val in enumerate(individual))

def show_results():
    root = tk.Tk()
    root.title("Genetic Algorithm Results")
    formatted_best_individual = format_best_individual(best_individual)
    tk.Label(root, text=f"Best paired Solution: {formatted_best_individual}").pack(pady=10, padx=10)
    tk.Label(root, text=f"Best Solution: {best_individual}").pack(pady=10, padx=10)
    tk.Label(root, text=f"Best Fitness Score: {best_score}").pack(pady=10)
    tk.Button(root, text="Close", command=root.destroy).pack(pady=20)
    root.mainloop()

show_results()


# Tracks and plots the maximum, minimum, and average fitness over all generations,
# providing a graphical representation of the algorithm's performance.
if fitness_history:
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_generations), fitness_history, label='Max Fitness')
    plt.plot(range(num_generations), min_fitness_history, label='Min Fitness')
    plt.plot(range(num_generations), avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution Over Generations')
    plt.ylim(30, 75)
    plt.legend()
    plt.show()
else:
    print("No valid generations to plot.")

# Gather maximum and minimum values from the entire fitness history for normalization
# overall_max = max(fitness_history)
# overall_min = min(fitness_history)
#
# # Normalize the fitness histories
# normalized_fitness_history = [(f - overall_min) / (overall_max - overall_min) for f in fitness_history]
# normalized_min_fitness_history = [(f - overall_min) / (overall_max - overall_min) for f in min_fitness_history]
# normalized_avg_fitness_history = [(f - overall_min) / (overall_max - overall_min) for f in avg_fitness_history]
#
# # Plot normalized fitness evolution
# plt.figure(figsize=(10, 5))
# plt.plot(range(num_generations), normalized_fitness_history, label='Max Fitness')
# plt.plot(range(num_generations), normalized_min_fitness_history, label='Min Fitness')
# plt.plot(range(num_generations), normalized_avg_fitness_history, label='Average Fitness')
# plt.xlabel('Generation')
# plt.ylabel('Normalized Fitness')
# plt.title('Normalized Fitness Evolution Over Generations')
# plt.legend()
# plt.show()
#print(population)
#print(len(population))

print(population_size)
