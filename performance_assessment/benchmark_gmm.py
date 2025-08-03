import os, sys
from sklearn.mixture import GaussianMixture
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lsh_density.lsh_based_kde import compute_densities_dataset

def sample_gmm(n_samples: int = 10000, n_features: int = 50, n_components: int = 3, random_state=None):
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
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    X = np.random.randn(n_samples, n_features)
    gmm.fit(X)
    samples, _ = gmm.sample(n_samples)
    densities = np.exp(gmm.score_samples(samples)).tolist()
    return samples, densities


def run_gmm_benchmark(
        n_samples: int = 10000, 
        n_features: int = 50, 
        n_components: int = 3, 
        n_hash_planes: int = 15,
        n_hash_tables: int = 30,
        random_state=None
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
    samples, gmm_densities = sample_gmm(
        n_samples=n_samples, 
        n_features=n_features, 
        n_components=n_components, 
        random_state=random_state
    )
    lsh_densities = compute_densities_dataset(
        dataset=samples, 
        hash_planes=n_hash_planes, 
        hash_tables=n_hash_tables
    )
    return gmm_densities, lsh_densities

