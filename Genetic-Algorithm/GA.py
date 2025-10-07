import random
import math

class SimpleGeneticAlgorithm:
    def __init__(self):
        
        self.target_value = 30
        self.number_of_chromosomes = 10  # No of solutions keept in each generation
        self.chromosomes = []  # List current solutions
        self.fitness_scores = []  # HGoodness each solution is (lower = better)
        self.fitness_percentages = []  # Probability of selecting each solution
        self.current_generation = 0
        
    def create_initial_population(self):
        """Create the first generation of random solutions"""
        print("="*60)
        print("GENETIC ALGORITHM SOLVER")
        print("Solving: a + 2b + 3c + 4d = 30")
        print("="*60)
        
        self.chromosomes = []
        for i in range(self.number_of_chromosomes):
            chromosome = [random.randint(1, 29) for gene in range(4)]
            self.chromosomes.append(chromosome)
        
        self.calculate_fitness_scores()
        
        self.show_chromosome_table()
        
    def calculate_fitness_scores(self):
        """Calculate how close each chromosome is to our target"""
        self.fitness_scores = []
        
        for chromosome in self.chromosomes:
            a, b, c, d = chromosome
            result = a + 2*b + 3*c + 4*d  # Calculate the equation result
            
            difference = result - self.target_value
            if difference < 0:
                fitness = -difference  
            else:
                fitness = difference   
            
            self.fitness_scores.append(fitness)
        
        
        total_inverse_fitness = 0
        for fitness in self.fitness_scores:
            if fitness == 0: 
                total_inverse_fitness += 1000
            else:
                total_inverse_fitness += 1 / (fitness ** 2)  # Inverse 
        
        self.fitness_percentages = []
        for fitness in self.fitness_scores:
            if fitness == 0:  
                percentage = (1000 / total_inverse_fitness) * 100
            else:
                percentage = ((1 / (fitness ** 2)) / total_inverse_fitness) * 100
            self.fitness_percentages.append(percentage)
    
    def show_chromosome_table(self):
        """Display all chromosomes in a nice table format"""
        print("\nCHROMOSOME TABLE:")
        print("="*80)
        print(f"{'Chromosome':<12} {'(a,b,c,d)':<15} {'Fitness':<20} {'Fitness %':<15}")
        print("-"*80)
        
        # Show each chromosome
        for i in range(len(self.chromosomes)):
            chromosome = self.chromosomes[i]
            fitness = self.fitness_scores[i]
            percentage = self.fitness_percentages[i]
            
            a, b, c, d = chromosome
            equation_result = a + 2*b + 3*c + 4*d
            fitness_calculation = f"|{equation_result}-30| = {fitness}"
            
            print(f"{i+1:<12} {str(tuple(chromosome)):<15} {fitness_calculation:<20} {percentage:.2f}%")
        
        print("-"*80)
        
        perfect_solution_found = False
        for i in range(len(self.fitness_scores)):
            fitness = self.fitness_scores[i]
            percentage = self.fitness_percentages[i]
            
            if fitness == 0 or percentage >= 99.0:
                print(f"PERFECT SOLUTION FOUND!")
                print(f"Chromosome {i+1}: {self.chromosomes[i]} with fitness {fitness} ({percentage:.2f}%)")
                a, b, c, d = self.chromosomes[i]
                print(f"Verification: {a} + 2*{b} + 3*{c} + 4*{d} = {a + 2*b + 3*c + 4*d}")
                perfect_solution_found = True
                break
        
        if not perfect_solution_found:
            best_chromosome_index = self.fitness_percentages.index(max(self.fitness_percentages))
            best_percentage = self.fitness_percentages[best_chromosome_index]
            best_fitness = self.fitness_scores[best_chromosome_index]
            print(f"BEST: Chromosome {best_chromosome_index + 1} with {best_percentage:.2f}% fitness rate (fitness: {best_fitness})")
        
        print()
        return perfect_solution_found
    
    def ask_user_what_to_do_next(self):
        """Ask the user what genetic operation they want to perform"""
        print("What would you like to do next?")
        print("1. Crossover")
        print("2. Mutation")
        print("3. Exit")
        
        while True:
            user_choice = input("\nEnter your choice (1-3): ").strip()
            if user_choice in ['1', '2', '3']:
                return user_choice
            else:
                print("Please enter 1, 2, or 3")
    
    def do_crossover(self):
        """Create new chromosomes by combining existing ones"""
        print("\nCROSSOVER OPERATION")
        print("="*50)
        print("Using roulette wheel selection based on fitness percentages...")
        print("Generating new offspring chromosomes...\n")
        
        new_chromosomes = []
        
        # new offspring
        for offspring_number in range(self.number_of_chromosomes):
            # Select two parents
            parent1_index = self.select_parent_using_roulette_wheel()
            parent2_index = self.select_parent_using_roulette_wheel()
            
            attempts = 0
            while parent2_index == parent1_index and attempts < 5:
                parent2_index = self.select_parent_using_roulette_wheel()
                attempts += 1
            
            parent1 = self.chromosomes[parent1_index].copy()
            parent2 = self.chromosomes[parent2_index].copy()
            
            crossover_point = random.randint(1, 3)
            
            offspring = parent1[:crossover_point] + parent2[crossover_point:]
            new_chromosomes.append(offspring)
            
            parent1_percentage = self.fitness_percentages[parent1_index]
            parent2_percentage = self.fitness_percentages[parent2_index]
            print(f"Offspring:")
            print(f"  Parents: Chromosome {parent1_index + 1} ({parent1_percentage:.1f}%) {parent1}")
            print(f"           Chromosome {parent2_index + 1} ({parent2_percentage:.1f}%) {parent2}")
            print(f"  Crossover point: {crossover_point}")
            print(f"  Result: {offspring}\n")
        
        best_chromosome_index = self.fitness_scores.index(min(self.fitness_scores))
        best_chromosome = self.chromosomes[best_chromosome_index]
        new_chromosomes[0] = best_chromosome

        self.chromosomes = new_chromosomes
        self.current_generation += 1
        
        print(f"--- Generation {self.current_generation} (After Crossover) ---")
        self.calculate_fitness_scores()
        self.show_chromosome_table()
    
    def select_parent_using_roulette_wheel(self):
        """Select a parent based on fitness percentages (better = more likely to be selected)"""
        random_number = random.uniform(0, 100)
        cumulative_percentage = 0
        
        for i in range(len(self.fitness_percentages)):
            cumulative_percentage += self.fitness_percentages[i]
            if random_number <= cumulative_percentage:
                return i
        
        return len(self.chromosomes) - 1
    
    def do_mutation(self):
        """Randomly change some genes in some chromosomes"""
        print("\nMUTATION OPERATION")
        print("="*50)
        print("Mutating chromosomes with small changes...\n")
        
        for chromosome_index in range(self.number_of_chromosomes):
            if random.random() < 0.4:
                gene_position = random.randint(0, 3)
                old_value = self.chromosomes[chromosome_index][gene_position]
                
                change = random.choice([-2, -1, 1, 2])
                new_value = old_value + change
                
                new_value = max(1, min(29, new_value))
                
                self.chromosomes[chromosome_index][gene_position] = new_value
                print(f"Chromosome {chromosome_index+1}: position {gene_position} changed from {old_value} to {new_value}")
        
        best_chromosome_index = self.fitness_scores.index(min(self.fitness_scores))
        best_chromosome = self.chromosomes[best_chromosome_index]
        self.chromosomes[0] = best_chromosome
        
        self.current_generation += 1
        print(f"\n--- Generation {self.current_generation} (After Mutation) ---")
        self.calculate_fitness_scores()
        self.show_chromosome_table()

def main():
    """Main function that runs the genetic algorithm"""
    genetic_algorithm = SimpleGeneticAlgorithm()
    genetic_algorithm.create_initial_population()
    
    while True:
        perfect_solution_exists = any(fitness == 0 for fitness in genetic_algorithm.fitness_scores)
        near_perfect_solution_exists = any(percentage >= 99.0 for percentage in genetic_algorithm.fitness_percentages)
        
        if perfect_solution_exists or near_perfect_solution_exists:
            print("Perfect solution achieved!")
            continue_experimenting = input("Continue experimenting? (y/n): ").lower()
            if continue_experimenting != 'y':
                break
        
        user_choice = genetic_algorithm.ask_user_what_to_do_next()
        
        if user_choice == '1':
            genetic_algorithm.do_crossover()
        elif user_choice == '2':
            genetic_algorithm.do_mutation()
        elif user_choice == '3':
            print("Goodbye!")
            break

# Run 
if __name__ == "__main__":
    main()
