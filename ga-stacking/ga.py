"""
Module: ga
----------
Genetic Algorithm for optimizing weight vectors over meta-features.
"""
import numpy as np
import random
from sklearn.metrics import roc_auc_score, f1_score


def initialize_population(pop_size: int, n_models: int) -> list:
    """
    Initialize population of normalized weight vectors.
    """
    pop = []
    for _ in range(pop_size):
        w = np.random.rand(n_models)
        w /= w.sum()
        pop.append(w)
    return pop


def evaluate(
    population: list,
    meta_X: np.ndarray,
    y_true: np.ndarray,
    metric: str = 'auc'
) -> tuple:
    """
    Evaluate each weight vector by computing ensemble metric.

    Args:
        population: list of weight arrays
        meta_X    : array (n_samples, n_models)
        y_true    : array (n_samples,)
        metric    : 'auc' or 'f1'

    Returns:
        scores    : list of fitness scores sorted desc
        sorted_pop: corresponding sorted population
    """
    scores = []
    for w in population:
        w = w / w.sum()
        ens = np.dot(meta_X, w)
        if metric == 'auc':
            scores.append(roc_auc_score(y_true, ens))
        else:
            scores.append(f1_score(y_true, ens > 0.5))
    idx = np.argsort(scores)[::-1]
    return [scores[i] for i in idx], [population[i] for i in idx]


def selection(population: list, scores: list) -> list:
    """Elitism + tournament selection."""
    next_pop = [population[0]]
    while len(next_pop) < len(population):
        candidates = random.sample(list(zip(population, scores)), k=4)
        winner = max(candidates, key=lambda x: x[1])[0]
        next_pop.append(winner)
    return next_pop


def crossover(population: list, prob: float) -> list:
    """Blend crossover on real-value vectors."""
    next_pop = [population[0]]
    i = 1
    while i < len(population):
        if random.random() < prob and i+1 < len(population):
            p1, p2 = population[i], population[i+1]
            alpha = random.random()
            c1 = alpha*p1 + (1-alpha)*p2
            c2 = alpha*p2 + (1-alpha)*p1
            next_pop += [c1/c1.sum(), c2/c2.sum()]
            i += 2
        else:
            next_pop.append(population[i])
            i += 1
    while len(next_pop) < len(population):
        next_pop.append(random.choice(population))
    return next_pop


def mutation(population: list, prob: float, scale: float = 0.1) -> list:
    """Gaussian mutation with renormalization."""
    next_pop = [population[0]]
    for ind in population[1:]:
        for j in range(len(ind)):
            if random.random() < prob:
                ind[j] += np.random.normal(0, scale)
        ind[ind < 0] = 1e-4
        ind /= ind.sum()
        next_pop.append(ind)
    return next_pop


def GA_weighted(
    meta_X_train: np.ndarray,
    y_train: np.ndarray,
    meta_X_val: np.ndarray,
    y_val: np.ndarray,
    pop_size: int = 50,
    generations: int = 30,
    crossover_prob: float = 0.6,
    mutation_prob: float = 0.3,
    metric: str = 'auc',
    verbose: bool = True
) -> np.ndarray:
    """
    Main GA loop, returns best weight vector.
    """
    n_models = meta_X_train.shape[1]
    population = initialize_population(pop_size, n_models)
    
    best_scores = []  # Track best score per generation for convergence analysis
    
    for gen in range(generations):
        scores, population = evaluate(population, meta_X_val, y_val, metric)
        population = selection(population, scores)
        population = crossover(population, crossover_prob)
        population = mutation(population, mutation_prob)
        
        best_scores.append(scores[0])
        
        if verbose:
            print(f'Gen {gen+1}/{generations} - best {metric}: {scores[0]:.4f}')
    
    return population[0], best_scores