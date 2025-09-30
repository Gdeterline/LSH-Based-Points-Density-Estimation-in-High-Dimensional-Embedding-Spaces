import math
import numpy as np
from collections import defaultdict
from itertools import combinations

def ready_hash_tables_functions(K: int, D: int, L: int, random_state: int = None):
    """
    Prepares the hash tables and hash planes for LSH-based KDE.
    
    Parameters
    ----------
    - K: int
        Number of hash bits per table (defines the number of buckets: 2^K).
    - D: int
        Number of dimensions in the dataset.
    - L: int
        Number of hash tables to use.
    - random_state: int, optional
        Random seed for reproducibility. If None, no seed is set.
        
    Returns
    -------
    - hash_planes: list of np.array
        List of hash planes, each of shape (K, D).
    - hash_tables: list of defaultdict
        List of hash tables, each a defaultdict(int) to count occurrences of hash keys.
    """
    if random_state is not None:
        np.random.seed(random_state) # for reproducibility
    hash_planes = [np.random.randn(K, D) for _ in range(L)]
    hash_tables = [defaultdict(list) for _ in range(L)]
    return hash_planes, hash_tables

def simhash(vector, planes):
    """
    Computes the SimHash for a given vector using the provided hash planes.

    Parameters
    ----------
    - vector: np.array
        The input vector to hash.
    - planes: np.array
        The hash planes to use for hashing, of shape (K, D) where K is the number of hash bits and D is the dimensionality of the vector.
    
    Returns
    -------
    - hash_key: tuple
        The computed hash key, a tuple of length K where each element is a boolean indicating \
        whether the dot product with the corresponding plane is positive.
    """
    # hash key (ex: key = (0, 1, 1, 1, 0, 0, ..., 1), len(key) = K)
    hash_key = tuple((planes @ vector) > 0)
    return hash_key

def hash_dataset(dataset, hash_planes, hash_tables):
    """
    Hashes the dataset using the provided hash planes and updates the hash tables.
    
    Parameters
    ----------
    - dataset: np.array
        The input dataset, an array of shape (n_samples, n_features).
    - hash_planes: list of np.array
        List of hash planes, each of shape (K, D) where K is the number of hash bits and D is the dimensionality of the dataset.
    - hash_tables: list of defaultdict
        List of hash tables, each a defaultdict(int) to count occurrences of hash keys.
    
    Returns
    -------
    - None
        The function updates the hash tables in place.
    """
    for x in dataset:
        for l in range(len(hash_tables)):
            h = simhash(x, hash_planes[l])
            hash_tables[l][h].append(x)

def run_lsh_clustering(dataset: np.array, hash_bits_per_table: int, number_of_hash_tables: int, random_state: int = None, verbose: bool = True) -> list:
    """
    Runs LSH-based clustering on the input dataset.
    
    Parameters
    ----------
    - dataset: np.array
        The input dataset, an array of shape (n_samples, n_features).
    - hash_bits_per_table: int
        Number of hash bits per table (defines the number of buckets: 2^K).
    - number_of_hash_tables: int
        Number of hash tables to use.
    - random_state: int, optional
        Random seed for reproducibility. If None, no seed is set.
    - verbose: bool, optional
        If True, prints progress messages.

    Returns
    -------
    - clusters: list
        A list of clusters, where each cluster is a list of data points.
    """
    if verbose:
        print("Preparing hash tables and planes...")

    hash_planes, hash_tables = ready_hash_tables_functions(K=hash_bits_per_table, D=dataset.shape[1], L=number_of_hash_tables, random_state=random_state)

    hash_dataset(dataset, hash_planes, hash_tables)
    
    if verbose:
        print("Hashing completed. Extracting clusters...")
    
    # n_samples = dataset.shape[0]
    # majority_threshold = number_of_hash_tables // 2 + 1

    # # Build per-table buckets as indices
    # buckets_idx_per_table = []
    # for l in range(number_of_hash_tables):
    #     signs = (dataset @ hash_planes[l].T) > 0  # shape (n_samples, hash_bits_per_table)
    #     table = defaultdict(list)
    #     for idx, key in enumerate(map(tuple, signs)):
    #         table[key].append(idx)
    #     buckets_idx_per_table.append(table)

    # # Count co-collisions across tables
    # pair_counts = defaultdict(int)
    # for table in buckets_idx_per_table:
    #     for idxs in table.values():
    #         if len(idxs) < 2:
    #             continue
    #         for i, j in combinations(idxs, 2):
    #             if i > j:
    #                 i, j = j, i
    #             pair_counts[(i, j)] += 1

    # # Union-Find to form clusters where pairs collide in a majority of tables
    # parent = list(range(n_samples))
    # rank = [0] * n_samples

    # def find(a: int) -> int:
    #     while parent[a] != a:
    #         parent[a] = parent[parent[a]]
    #         a = parent[a]
    #     return a

    # def union(a: int, b: int) -> None:
    #     ra, rb = find(a), find(b)
    #     if ra == rb:
    #         return
    #     if rank[ra] < rank[rb]:
    #         parent[ra] = rb
    #     elif rank[ra] > rank[rb]:
    #         parent[rb] = ra
    #     else:
    #         parent[rb] = ra
    #         rank[ra] += 1

    # for (i, j), c in pair_counts.items():
    #     if c >= majority_threshold:
    #         union(i, j)

    # # Assign integer labels and build clusters
    # root_to_label = {}
    # clusters_dict = defaultdict(list)
    # next_label = 0
    # for idx in range(n_samples):
    #     r = find(idx)
    #     if r not in root_to_label:
    #         root_to_label[r] = next_label
    #         next_label += 1
    #     clusters_dict[root_to_label[r]].append(dataset[idx])

    # clusters = [clusters_dict[i] for i in range(next_label)]

    # if verbose:
    #     print(f"Formed {len(clusters)} clusters using majority threshold {majority_threshold}/{number_of_hash_tables}.")

    # return clusters


def clustering_granularity(n_bits: int = 16, p_target: float = 0.5):
    """
    Computes the granularity of the clustering given the number of bits and target probability.
    The higher the number of bits, the finer the granularity of the clustering (because more buckets are created, leading to smaller clusters).
    The granularity is defined as the angle θ such that the cosine of θ gives the threshold for the cosine similarity.
    The relationship is derived from the equation (1 - θ/π)^n_bits = p_target.
    
    Parameters
    ----------
    - n_bits: int = 16
        The number of bits used in the LSH hash function.
    - p_target: float = 0.5
        The target probability for the cosine similarity threshold. Default is 0.5.
        
    Returns
    -------
    - theta: float
        The angle threshold θ in radians.
    - cosine_threshold: float
        The cosine threshold corresponding to the angle θ.
    """
    theta = math.pi * (1 - p_target ** (1 / n_bits))
    cosine_threshold = math.cos(theta)
    return theta, cosine_threshold