# Soft Computing Project: Genetic Algorithm for TSP

This project demonstrates the use of **Genetic Algorithms (GAs)** to solve the **Traveling Salesman Problem (TSP)**, implemented as part of a Soft Computing course. The algorithm efficiently finds near-optimal solutions for visiting a set of 15 cities exactly once and returning to the starting city.


## 🧠 Key Concepts Used

- **Genetic Algorithm (GA)**
- **Roulette Wheel Selection**
- **One-Point Crossover**
- **Inorder Swap Mutation**
- **Elitism**
- **Chromosome Rectification**

## 🧮 Problem Description

> Find the shortest possible route that visits each of 15 cities exactly once and returns to the origin city.

The TSP is a classic NP-hard problem. This implementation uses a GA to approximate the optimal solution.

---

## ⚙️ Methods and Operators Used

### 1. **Initialization**
- Population of 15 chromosomes (routes)
- Each chromosome is a random permutation of cities `[0, 1, ..., 14]`

### 2. **Fitness Function**
- Fitness = `1 / Total Distance`  
- Encourages shorter routes (i.e., higher fitness)

### 3. **Selection**
- **Roulette Wheel Selection**:
  - Selects parents with probability proportional to their fitness

### 4. **Crossover**
- **One-Point Crossover**:
  - Random split point
  - First half of Parent A + second half of Parent B
  - Rectification ensures valid chromosome (no repeated cities)

### 5. **Mutation**
- **Inorder Swap Mutation**:
  - Randomly selects and swaps two cities in a chromosome

### 6. **Elitism**
- The best solution in a generation is preserved by replacing the worst one in the next generation

---

## 🛠️ Implementation

The core algorithm is encapsulated in a Python class `GeneticAlgorithmTSP`. Key methods include:

- `initialize_population()`
- `calculate_fitness()`
- `roulette_wheel_selection()`
- `crossover()`
- `mutate()`
- `rectify_chromosome()`
- `run()`

You can either input a custom 15×15 distance matrix or let the program generate one randomly.

---

## 📥 Input

- **Option 1**: Randomly generated symmetric 15x15 matrix with zero diagonals
- **Option 2**: Manually input 15 rows of 15 space-separated integers

---

## 📤 Output

- Best route found after 20 iterations
- Total distance of that route
- Route printed in order of city traversal

---

## ▶️ Running the Program

```bash
python problem2_TSP.py
