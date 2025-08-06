import numpy as np
from collections import defaultdict


def ready_hash_tables_functions(K: int, D: int, L: int, random_state: int = None):
    if random_state is not None:
        np.random.seed(random_state) # for reproducibility
    hash_planes = [np.random.randn(K, D) for _ in range(L)]
    hash_tables = [defaultdict(int) for _ in range(L)]
    return hash_planes, hash_tables

def simhash(vector, planes):
    # hash key (ex: key = (0, 1, 1, 1, 0, 0, ..., 1), len(key) = K)
    return tuple((planes @ vector) > 0)

def hash_dataset(dataset, hash_planes, hash_tables):
    for x in dataset:
        for l in range(len(hash_tables)):
            h = simhash(x, hash_planes[l])
            hash_tables[l][h] += 1

def estimate_bucket_count(query, hash_planes, hash_tables):
    query = query / np.linalg.norm(query)
    counts = []
    for l in range(len(hash_tables)):
        h = simhash(vector=query, planes=hash_planes[l])
        count = hash_tables[l].get(h, 0)
        counts.append(count)
    return np.mean(counts)

def compute_relative_densities(bucket_counts):
    """
    Computes the relative densities from the bucket counts.
    
    Parameters
    ----------
    - bucket_counts: list of int
        List of bucket counts for each query point.
        
    Returns
    -------
    - relative_densities: list of float
        List of relative densities for each query point.
    """
    total_count = len(bucket_counts)
    assert total_count > 0, "The length of bucket_counts must be greater than 0."
    densities = [count / total_count for count in bucket_counts]
    return densities

def compute_bucket_count_dataset(dataset, hash_planes, hash_tables):
    bucket_counts = []
    queries = dataset 
    for i, q in enumerate(queries):
        bucket_counts.append(estimate_bucket_count(query=q, hash_planes=hash_planes, hash_tables=hash_tables) - 1)  # -1 to exclude the query point itself
    return bucket_counts



# def check_bucket_distribution(hash_tables):
#     all_counts = []
#     for table in hash_tables:
#         all_counts.extend(table.values())
#     return all_counts


def run_high_dim_kde(dataset: np.array, hash_bits_per_table: int, number_of_hash_tables: int, random_state: int = None, verbose: bool = True) -> list:
    """
    Computes the estimated points density across the dataset in high dimensional space.

    Parameters
    ----------

    - dataset: np.array[np.array]
        array of shape (nb_samples, nb_dimensions)
        dataset input may be composed of normalized or not vectors

    - hash_bits_per_table: int
        The number of hash bits per table defines the number of buckets: 2^n

    - number_of_hash_tables: int
        The number of hash tables
        
    - verbose: bool = True
        Optional parameter to control verbosity of the function.
        Default is True.
        If True, prints additional information during the computation

    Returns
    -------
    Returns the list of points density estimates

    """
    if verbose:
        print(f"Running LSH-based KDE with {hash_bits_per_table} hash bits per table and {number_of_hash_tables} hash tables.")
    if dataset.shape[0] == 0:
        raise ValueError("The dataset is empty. Please provide a valid dataset with samples.")
    if dataset.shape[1] == 0:
        raise ValueError("The dataset has no features. Please provide a valid dataset with features.")
    if hash_bits_per_table <= 0:
        raise ValueError("The number of hash bits per table must be a positive integer.")
    if number_of_hash_tables <= 0:
        raise ValueError("The number of hash tables must be a positive integer.")
    if verbose:
        print("Preparing hash tables and planes...")
    hash_planes, hash_tables = ready_hash_tables_functions(K=hash_bits_per_table, D=dataset.shape[1], L=number_of_hash_tables, random_state=random_state)
    if verbose:
        print("Hash tables and planes prepared.")
        print("Hashing the dataset...")
    hash_dataset(dataset, hash_planes, hash_tables)
    if verbose:
        print("Dataset hashed.")
        print("Computing densities for the dataset...")
    bucket_counts = compute_bucket_count_dataset(dataset=dataset, hash_planes=hash_planes, hash_tables=hash_tables)
    densities = compute_relative_densities(bucket_counts=bucket_counts)
    if len(densities) != dataset.shape[0]:
        raise ValueError("The length of densities does not match the number of samples in the dataset.")
    if verbose:
        print("Densities computed.")
        print(f"Computed densities using LSH and SimHash for {len(densities)} samples.")
    return densities