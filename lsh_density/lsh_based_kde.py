import numpy as np
from collections import defaultdict


def ready_hash_tables_functions(K, D, L):
    np.random.seed(10) # for reproducibility
    hash_planes = [np.random.randn(K, D) for _ in range(L)]
    hash_tables = [defaultdict(int) for _ in range(L)]
    return hash_planes, hash_tables

def simhash(vec, planes):
    # hash key (ex: key = (0, 1, 1, 1, 0, 0, ..., 1), len(key) = K)
    return tuple((planes @ vec) > 0)

def hash_dataset(dataset, hash_planes, hash_tables):
    for x in dataset:
        for l in range(len(hash_tables)):
            h = simhash(x, hash_planes[l])
            hash_tables[l][h] += 1

def estimate_density(query, hash_planes, hash_tables):
    query = query / np.linalg.norm(query)
    counts = []
    for l in range(len(hash_tables)):
        h = simhash(query, hash_planes[l])
        count = hash_tables[l].get(h, 0)
        counts.append(count)
    return np.mean(counts)

def compute_densities_dataset(dataset, hash_planes, hash_tables):
    densities = []
    queries = dataset 
    for i, q in enumerate(queries):
        densities.append(estimate_density(q, hash_planes, hash_tables))
    return densities

# def check_bucket_distribution(hash_tables):
#     all_counts = []
#     for table in hash_tables:
#         all_counts.extend(table.values())
#     return all_counts


def run_high_dim_kde(dataset: np.array, hash_bits_per_table: int, number_of_hash_tables: int) -> list:
    """
    Computes the estimated points density across the dataset in high dimensional space.

    Parameters
    ----------

    dataset: np.array[np.array]
    array of shape (nb_samples, nb_dimensions)
    dataset input may be composed of normalized or not vectors

    hash_bits_per_table: int
    The number of hash bits per table defines the number of buckets: 2^n

    number_of_hash_tables: int
    The number of hash tables

    Returns
    -------
    Returns the list of points density estimates

    """
    hash_planes, hash_tables = ready_hash_tables_functions(hash_bits_per_table, dataset.shape[1], number_of_hash_tables)
    hash_dataset(dataset, hash_planes, hash_tables)
    densities = compute_densities_dataset(dataset, hash_planes, hash_tables)
    return densities