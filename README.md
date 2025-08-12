# LSH-Based Points Density Estimation in High-Dimensional Embedding Spaces
 
This project implements a fast and scalable point density estimator using **Locality Sensitive Hashing (LSH)** with **SimHash**, designed for high-dimensional spaces. The estimator is particularly useful for applications like clustering, anomaly detection, and data visualization in large datasets.

We will benchmark its accuracy and runtime against **Kernel Density Estimation (KDE)** and **k-NN density estimation** using synthetic data generated from a **Gaussian Mixture Model (GMM)** with known ground-truth density.

---

## Table of Contents

- [I. Introduction](#introduction)
- [II. Features](#features)
- [III. Installation](#installation)
- [IV. Documentation](#documentation)
  - [IV.1. Theoretical Concepts](#theoretical-concepts)
    - [IV.1.1. Theoretical Explanation of Density Estimation with LSH and SimHash](#theoretical-explanation-of-density-estimation-with-lsh-and-simhash)
    - [IV.1.2. Mathematical Formulation](#mathematical-formulation)
- [V Implementation Details](#implementation-details)
- [VI. Performance Assessment](#performance-assessment)
    - [VI.1. Synthetic Data Generation](#synthetic-data-generation-1)
    - [VI.2. Plotting the Density Estimates](#plotting-the-density-estimates)
    - [VI.3. Quantitative Metrics](#quantitative-metrics)
- [VII. Benchmarking](#benchmarking)
  - [VII.1. Benchmarking against KDE and k-NN](#benchmarking-against-kde-and-k-nn)
- [VIII. Usage](#usage)
- [IX. Conclusion](#conclusion)
- [X. License](#license)
- [XI. Contact](#contact)

---

## I. Introduction
In high-dimensional spaces, traditional density estimation methods like Kernel Density Estimation (KDE) and k-NN can struggle with performance and accuracy due to the curse of dimensionality. Indeed, as the number of dimensions increases, data becomes increasingly sparse, and these methods require exponentially more samples to produce reliable estimates. This sparsity leads to poor generalization and high computational costs, especially in deep embedding spaces where data is represented by high-dimensional vectors learned by neural networks.
To address these challenges, we propose a scalable density estimation method leveraging Locality Sensitive Hashing (LSH) with SimHash, which efficiently approximates point densities by grouping similar vectors into hash buckets, thus enabling fast, accurate and scalable density estimation in deep embedding spaces.


---

## II. Features
- **Locality Sensitive Hashing (LSH)**: Utilizes LSH with SimHash to efficiently group similar points in high-dimensional spaces.
- **SimHash**: Implements SimHash for hashing high-dimensional vectors, enabling fast similarity search.
- **Density Estimation**: Estimates point density in high-dimensional spaces using LSH, providing a scalable alternative to traditional methods. Handles the curse of dimensionality effectively.
- **Benchmarking**: Compares the performance of the LSH-based density estimator against traditional methods like KDE and k-NN.
- **Synthetic Data Generation**: Uses GMM to create datasets with known density distributions for accurate benchmarking.
- **Visualization**: Provides tools to visualize density estimates and compare methods.
- **Documentation**: Includes detailed documentation and examples for easy usage.

## III. Installation


---

## IV. Documentation

### IV.1. Theoretical Concepts
This section provides an overview of the theoretical concepts behind the LSH-based point density estimator, including Locality Sensitive Hashing (LSH), SimHash, and density estimation techniques.

#### IV.1.1 Theoretical Explanation of Density Estimation with LSH and SimHash

The key idea enabling density estimation here is that SimHash is a locality-sensitive hash function for cosine similarity. 
This means that similar vectors (in terms of cosine similarity) are more likely to hash to the same bucket across multiple hash tables. Indeed, raw inputs that are semantically close to each other will yield similar embeddings, and thus similar hash keys in the hash tables.
Buckets in the hash tables will contain counts of how many vectors hash to that bucket, which can be interpreted as a density estimate for that region of the embedding space.
The density estimate from one hash table counts how many points fall into the same bucket — therefore it approximates the local neighborhood density.
But given hash buckets may contain dissimilar points (false positives), a single table's estimate can be noisy.
To mitigate this noise, we use multiple independent hash tables. Each table provides a count of how many points fall into the same bucket, and we average these counts across all tables to get a more robust density estimate.

#### IV.1.2 Mathematical Formulation

Given a dataset of points (vectors) $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\}$ where each $\mathbf{x}_i \in \mathbb{R}^D$:

1. **Random Hyperplanes**  

We sample $k$ random hyperplanes $\{\mathbf{W}_1, \mathbf{W}_2, \ldots, \mathbf{W}_k\}$, where each $\mathbf{W}_j \in \mathbb{R}^D$ is drawn independently from a standard normal distribution:

$$
\mathbf{W}_j \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_D), \quad j = 1, \ldots, k
$$

2. **Hash Function Definition**  

The hash function $h: \mathbb{R}^D \to \{0,1\}^k$ maps a vector $\mathbf{x}$ to a $k$-bit binary string defined by the sign of dot products with the hyperplanes:

$$
h(\mathbf{x}) = \big( h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_k(\mathbf{x}) \big)
$$
where each bit is computed as:
$$
h_j(\mathbf{x}) = 
\begin{cases}
1 & \text{if } \mathbf{W}_j \cdot \mathbf{x} \geq 0 \\
0 & \text{if } \mathbf{W}_j \cdot \mathbf{x} < 0
\end{cases}
$$

3. **Hash Bucket Assignment**  

The hash code $h(\mathbf{x})$ is used to assign the vector $\mathbf{x}$ to a bucket in a hash table. Each unique hash code corresponds to a bucket, and all vectors that produce the same hash code are grouped together in that bucket.


4. **Density Estimation**

The density estimate for a point $\mathbf{x}$ is computed as the average count of points in the buckets corresponding to its hash code across multiple hash tables:
$$
\hat{p}(\mathbf{x}) = \frac{1}{L} \sum_{t=1}^{L} \text{count}_l(h(\mathbf{x}))
$$
where $\text{count}_l(h(\mathbf{x}))$ is the number of points in the bucket corresponding to the hash code $h(\mathbf{x})$ in the $l$-th hash table, and $L$ is the total number of hash tables.

5. **Normalization**

Given that the hashing is based on the $\mathrm{sign}(x)$ function, theoretically, the vectors should not need to be normalized. Indeed, given that:

$$
h_j(\mathbf{x}) = 
\begin{cases}
1 & \text{if } \mathbf{W}_j \cdot \mathbf{x} \geq 0 \\
0 & \text{if } \mathbf{W}_j \cdot \mathbf{x} < 0
\end{cases}
$$

and 

$$
\mathbf{W}_j \cdot \mathbf{x} = ||\mathbf{W}_j|| \cdot ||\mathbf{x}|| \cdot \cos(\theta)
$$

where $\theta$ is the angle between the hyperplane and the vector.
Thus, given that $||\mathbf{W}_j||$ and $||\mathbf{x}||$ are both always positive, the sign of the dot product depends exclusively on the angle $\theta$ between the hyperplane and the vector. Therefore, normalization is not strictly necessary for the hashing process to work correctly.

6. **Example: Hashing three vectors**

For example, we can consider three vectors in 5-dimensional space. These three vectors will inclide two that are similar and one that is dissimilar. 

Let be the vectors:
$$
\mathbf{x}_1 = [1, 2, 3, 4, 5]$$
$$
\mathbf{x}_2 = [1.1, 2.1, 3.1, 4.1, 5.1]$$
$$
\mathbf{x}_3 = [53, 21, -42, -7, 2]$$
We can compute their hash codes using three random hyperplanes. The hyperplanes will be sampled from a standard normal distribution, and we will use them to compute the hash codes for each vector.

We can sample three random hyperplanes $\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3$ as follows:

$$
\mathbf{W}_1 = [0.5, -0.5, 0.5, -0.5, 0.5]$$
$$
\mathbf{W}_2 = [0.1, 0.2, 0.3, 0.4, 0.5]$$
$$
\mathbf{W}_3 = [-0.2, 0.3, -0.1, 0.4, -0.5]
$$

We can compute the hash codes for each vector using the sign of the dot product with the hyperplanes:

| Vector      | $\mathbf{W}_1 \cdot \mathbf{x}$ | $\mathbf{W}_2 \cdot \mathbf{x}$ | $\mathbf{W}_3 \cdot \mathbf{x}$ |
|-------------|:------------------------------:|:-------------------------------:|:-------------------------------:|
| $\mathbf{x}_1$ | $0.5*1 + (-0.5)*2 + 0.5*3 + (-0.5)*4 + 0.5*5 = 0.5 - 1 + 1.5 - 2 + 2.5 = 1.5$ | $0.1*1 + 0.2*2 + 0.3*3 + 0.4*4 + 0.5*5 = 0.1 + 0.4 + 0.9 + 1.6 + 2.5 = 5.5$ | $-0.2*1 + 0.3*2 + (-0.1)*3 + 0.4*4 + (-0.5)*5 = -0.2 + 0.6 - 0.3 + 1.6 - 2.5 = -0.8$ |
| $\mathbf{x}_2$ | $0.5*1.1 + (-0.5)*2.1 + 0.5*3.1 + (-0.5)*4.1 + 0.5*5.1 = 0.55 - 1.05 + 1.55 - 2.05 + 2.55 = 1.55$ | $0.1*1.1 + 0.2*2.1 + 0.3*3.1 + 0.4*4.1 + 0.5*5.1 = 0.11 + 0.42 + 0.93 + 1.64 + 2.55 = 5.65$ | $-0.2*1.1 + 0.3*2.1 + (-0.1)*3.1 + 0.4*4.1 + (-0.5)*5.1 = -0.22 + 0.63 - 0.31 + 1.64 - 2.55 = -0.81$ |
| $\mathbf{x}_3$ | $0.5*53 + (-0.5)*21 + 0.5*(-42) + (-0.5)*(-7) + 0.5*2 = 26.5 - 10.5 - 21 + 3.5 + 1 = 45$ | $0.1*53 + 0.2*21 + 0.3*(-42) + 0.4*(-7) + 0.5*2 = 5.3 + 4.2 - 12.6 - 2.8 + 1 = 35$ | $-0.2*53 + 0.3*21 + (-0.1)*(-42) + 0.4*(-7) + (-0.5)*2 = -10.6 + 6.3 + 4.2 - 2.8 - 1 = -30$ |

Given the dot products, we can compute the sign of each dot product to get the hash codes:

| Vector      | $\mathbf{W}_1 \cdot \mathbf{x}$ | $\mathbf{W}_2 \cdot \mathbf{x}$ | $\mathbf{W}_3 \cdot \mathbf{x}$ | Hash Code |
|-------------|:------------------------------:|:-------------------------------:|:-------------------------------:|:---------:|
| $\mathbf{x}_1$ | $1.5$ | $5.5$ | $-0.8$ | $(1, 1, 0)$ |
| $\mathbf{x}_2$ | $1.55$ | $5.65$ | $-0.81$ | $(1, 1, 0)$ |    
| $\mathbf{x}_3$ | $45$ | $35$ | $-30$ | $(1, 1, 1)$ |

The two similar vectors $\mathbf{x}_1$ and $\mathbf{x}_2$ hash to the same bucket $(1, 1, 0)$, while the dissimilar vector $\mathbf{x}_3$ hashes to a different bucket $(1, 1, 1)$.
This demonstrates how LSH with SimHash can effectively group similar vectors together while separating dissimilar ones, enabling efficient density estimation in high-dimensional spaces.


## V. Implementation Details

The implementation of the LSH-based point density estimator was made in Python, leveraging libraries like NumPy for efficient numerical computations and collections for managing hash tables. The code is structured to allow easy integration into existing data processing pipelines and can handle large datasets efficiently. 

The implementation includes the following key components:

**Parameters for LSH:**

```python
dataset : np.ndarray # Input dataset of shape (N, D) where N is the number of points and D is the embedding dimension
K : int  # hash bits per table -> nb_buckets = 2**K
L : int # number of hash tables
```

The parameters `K` and `L` control the number of bits per hash table and the number of hash tables, respectively. Increasing `K` increases the granularity of the hash buckets, while increasing `L` improves robustness against noise by averaging across multiple tables. Increasing these parameters will lead to more accurate density estimates but at the cost of increased memory usage and computation time.

**Building the hash functions and hash tables:**

The hash functions are built using random hyperplanes sampled from a standard normal distribution. Each hyperplane is used to compute the hash code for each point in the dataset, which is then stored in a hash table. This process is repeated for `L` hash tables, each with its own set of random hyperplanes.

The hash tables are implemented as dictionaries where keys are the hash codes and values are the number of points that hash to that code.

The hash tables and hash functions are built in the `ready_hash_tables_functions` function, which initializes the hash tables and hash functions.

The vectors are then hashed using the `hash_dataset` function, which computes the hash codes for each point in the dataset using the `simhash` function. The `simhash` function computes the hash code for a single point by taking the sign of the dot product with each hyperplane.

The computation of all densities is done in the `compute_densities_dataset` function, which iterates over each point in the dataset, calls in the `estimate_density` function to compute the density estimate for that point, and stores the result in a list.

## VI. Performance assessment

This section describes the benchmarking process used to evaluate the performance of the LSH-based point density estimator on synthetic data generated from a Gaussian Mixture Model (GMM). 
This will allow us to assess the "absolute" accuracy of the density estimates, by comparing them to the known ground-truth density of the GMM, through both qualitative and quantitative metrics.

### VI.1. Synthetic Data Generation

To evaluate the performance of the LSH-based point density estimator, we generated synthetic datasets using sklearn's `make_blobs` function, which creates isotropic Gaussian blobs for clustering. 
The datasets are each composed of several clusters, each of different sizes and densities, allowing us to assess the estimator's performance across varying conditions.

The initial performance assessment was conducted on a relatively small dataset of size within the range of 2500 to 10000 points, with embeddings of dimension 3.
Starting with 3 dimensional embeddings was chosen for two main reasons:
- First, it allows for easy visualization of the density estimates in 3D plots, which is useful for qualitative assessment.
- Second, it serves as a good starting point for benchmarking the performance of the LSH-based estimator. Given that the LSH-based estimator works with SimHash, 3D embeddings are the minimum dimensionality required to effectively demonstrate the locality-sensitive properties of the hashing function.

The following plot displays the results for such a dataset.

![Ground Truth vs LSH Density Estimate 3D plot comparison](/results/plots/ground_truth_vs_lsh_density_3d.png)

The left-hand side of the plot displays the 3D data points, colored by their ground truth densities.
The right-hand side of the plot displays the 3D data points, colored by their LSH-based density estimates.

The LSH density estimates per point seem accurate with respect to the other points (the colorscales), when comparing to the ground truth densities. Yet, it seems that the density values are different though within acceptable range. Depending on the use case, this can or cannot be problematic. This issue will be investigated further later on.

Then, we progressively increased the dataset size to ranges of 15000 to 30000 points, where each embedding is of dimension 50, to assess the scalability and performance of the estimator in higher-dimensional spaces.

#### VI.2. Plotting the Density Estimates

To visualize the performance of the LSH-based point density estimator, we plotted the estimated densities against the ground-truth densities of the synthetic dataset. The plots show how well the estimated densities match the true densities, providing a qualitative assessment of the estimator's accuracy.







---

## VII. License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## VIII. Contact

For any questions, suggestions, or contributions, please feel free to submit an issue or pull request on the GitHub repository. You can also reach out via [email](mailto:guillaumedt1001@gmail.com) or throuh my [LinkedIn](https://www.linkedin.com/in/guillaume-macquart-de-terline-a7b73430b) profile.
Thank you for your interest in this project! I hope it provides a valuable tool for density estimation in high-dimensional embedding spaces.