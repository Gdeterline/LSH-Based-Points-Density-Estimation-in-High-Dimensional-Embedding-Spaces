import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from scipy.stats import spearmanr, pearsonr


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lsh_density.lsh_based_kde import run_high_dim_kde

def sample_gmm(n_samples: int = 10000, n_features: int = 50, n_components: int = 3, random_state=None, verbose: bool = True):
    """
    Generate samples from a Gaussian Mixture Model (GMM).
    This function creates a GMM with specified parameters and generates samples from it.
    Also computes the density of each sample.

    Parameters
    ------------
    - n_samples: int = 10000
        number of samples to generate
        
    - n_features: int = 50
        number of features for each sample
    
    - n_components: int = 3
        number of Gaussian components in the mixture
    
    - random_state: int or None
        seed for reproducibility
    
    Returns
    ------------
    - samples: np.ndarray
        samples generated from the GMM
    
    - densities: list
        density values for each sample computed using the GMM
    """
    if verbose:
        print(f"Creating Gaussian Mixture Model with {n_components} components, "
              f"{n_features} features, and {n_samples} samples.")
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    X = np.random.randn(n_samples, n_features)
    if verbose:
        print(f"Fitting GMM to the data")
    gmm.fit(X)
    samples, _ = gmm.sample(n_samples)
    if verbose:
        print(f"Generated {n_samples} samples from GMM.")
        print(f"Computing densities for the generated samples")
    densities = np.exp(gmm.score_samples(samples)).tolist()
    if verbose:
        print(f"Computed densities for the samples.")
    return samples, densities

def sample_blobs(n_samples: int = 10000, n_features: int = 50, clusters_centers: int = 1, cluster_std: float = 1.0, random_state=None, verbose: bool = True):
    """
    This function creates a dataset of samples using the make_blobs function from sklearn.

    Parameters
    ------------
    - n_samples: int = 10000
        number of samples to generate
        
    - n_features: int = 50
        number of features for each sample
    
    - cluster_centers: int = 1
        number of clusters to generate
        
    - cluster_std: float = 1.0
        standard deviation of the clusters    
    
    - random_state: int or None
        seed for reproducibility
    
    - verbose: bool = True
        Optional parameter to control verbosity of the function.
        If True, prints additional information during the computation
    
    Returns
    ------------
    - samples: np.ndarray
        samples generated from the GMM
    
    - densities: list
        density values for each sample computed using the GMM
    """
    if verbose:
        print(f"Generating {n_samples} samples with {n_features} features each, "
              f"{clusters_centers} cluster centers, and standard deviation {cluster_std}.")
    
    X, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=clusters_centers, cluster_std=cluster_std, random_state=random_state)
    
    if verbose:
        print(f"Generated {n_samples} samples with {n_features} features each.")
    
    return X, labels.tolist()
    


def run_gmm_benchmark(
        n_samples: int = 10000, 
        n_features: int = 50, 
        n_components: int = 3, 
        n_hash_planes: int = 15,
        n_hash_tables: int = 30,
        random_state=None,
        verbose: bool = True
    ):
    """
    Run a benchmark comparing Gaussian Mixture Model (GMM) densities with LSH-based densities.
    This function generates samples from a GMM, computes their ground-truths densities, and computes LSH-based densities for the same samples.
    
    Parameters
    ------------
    - n_samples: int = 1000
        number of samples to generate
        
    - n_features: int = 50
        number of features for each sample
    
    - n_components: int = 3
        number of Gaussian components in the mixture
    
    - n_hash_planes: int = 15
        number of hash planes for LSH
    
    - n_hash_tables: int = 30
        number of hash tables for LSH
    
    - random_state: int or None
        seed for reproducibility
    
    Returns
    ------------
    - gmm_densities: list
        density values for each sample from the GMM
        
    - lsh_densities: list
        density values for each sample computed using LSH-based points density estimation

    """
    
    if verbose:
        print(f"Generating samples from GMM")
    
    start = time.time()
    
    samples, gmm_densities = sample_gmm(
        n_samples=n_samples, 
        n_features=n_features, 
        n_components=n_components, 
        random_state=random_state,
        verbose=verbose
    )
    
    if verbose:
        print(f"Generated {n_samples} samples with {n_features} features each from GMM with {n_components} components.")
    end = time.time()
    if verbose:
        print(f"Time taken to generate samples: {end - start:.2f} seconds")
        print(f"Computing LSH-based densities for the samples")
        
    start = time.time()
    
    lsh_densities = run_high_dim_kde(
        dataset=samples, 
        hash_bits_per_table=n_hash_planes, 
        number_of_hash_tables=n_hash_tables, 
        verbose=verbose
    )
    
    end = time.time()
    if verbose:
        print(f"Computed LSH-based densities for {n_samples} samples of {n_features} features each.")
        print(f"Time taken to compute LSH densities: {end - start:.2f} seconds")
    
    return gmm_densities, lsh_densities


def main(
    n_samples: int = 10000,
    n_features: int = 50,
    n_components: int = 3,
    n_hash_planes: int = 15,
    n_hash_tables: int = 30,
    random_state=None,
    verbose: bool = True
):
    """
    Main function to run the GMM benchmark.
    
    Parameters
    ------------
    - n_samples: int = 10000
        number of samples to generate
        
    - n_features: int = 50
        number of features for each sample
    
    - n_components: int = 3
        number of Gaussian components in the mixture
    
    - n_hash_planes: int = 15
        number of hash planes for LSH
    
    - n_hash_tables: int = 30
        number of hash tables for LSH
    
    - random_state: int or None
        seed for reproducibility
    """
    
    if verbose:
        print(f"Running GMM benchmark with parameters:\n"
              f"n_samples: {n_samples}, n_features: {n_features}, "
              f"n_components: {n_components}, n_hash_planes: {n_hash_planes}, "
              f"n_hash_tables: {n_hash_tables}, random_state: {random_state}")
        
    
    gmm_densities, bucket_count = run_gmm_benchmark(
        n_samples=n_samples, 
        n_features=n_features, 
        n_components=n_components, 
        n_hash_planes=n_hash_planes, 
        n_hash_tables=n_hash_tables,
        random_state=random_state,
        verbose=verbose
    )
    
    normalized_lsh_densities = [density / n_samples for density in bucket_count]
    
    if verbose:
        print(f"Benchmark completed. Plotting the results.")
    
    pearson_corr, _ = pearsonr(gmm_densities, bucket_count)
    spearman_corr, _ = spearmanr(gmm_densities, bucket_count)
    print(f"Pearson r = {pearson_corr:.3f}")
    print(f"Spearman rho = {spearman_corr:.3f}")
    
    # Plotting the results
    plt.figure(figsize=(12,6))
    plt.scatter(gmm_densities, normalized_lsh_densities, alpha=0.3, s=10)
    plt.xlabel("True GMM Density")
    plt.ylabel("Estimated LSH Density")
    plt.title("LSH vs. True Densities")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.hist(gmm_densities, bins=50, alpha=0.7, label='GMM Densities', color='blue')
    # plt.title('GMM Densities')
    # plt.xlabel('Density')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.hist(normalized_lsh_densities, bins=50, alpha=0.7, label='LSH Densities', color='orange')
    # plt.title('LSH Densities')
    # plt.xlabel('Density')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.tight_layout()
    # plt.suptitle(f'GMM vs LSH Density Estimation\n'
    #              f'Samples: {n_samples}, Features: {n_features}, Components: {n_components}, '
    #              f'Hash Planes: {n_hash_planes}, Hash Tables: {n_hash_tables}', fontsize=16)
    # plt.subplots_adjust(top=0.85)
    # plt.savefig('./results/plots/gmm_vs_lsh_density_estimation.png')
    # plt.show()
    
    
    
if __name__ == "__main__":
    main(
        n_samples=10000, 
        n_features=3, 
        n_components=1, 
        n_hash_planes=15, 
        n_hash_tables=30, 
        random_state=42
    )
    