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
  - [IV.2. Implementation Details](#implementation-details)
  - [IV.3. Benchmarking](#benchmarking)
    - [IV.2.1. Synthetic Data Generation](#synthetic-data-generation)
    - [IV.2.2. Benchmarking against KDE and k-NN](#benchmarking-against-kde-and-k-nn)
- [V. Usage](#usage)
- [VI. Conclusion](#conclusion)
- [VII. License](#license)
- [VIII. Contact](#contact)

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


### IV.2. Implementation Details

The implementation of the LSH-based point density estimator was made in Python, leveraging libraries like NumPy for efficient numerical computations and collections for managing hash tables. The code is structured to allow easy integration into existing data processing pipelines and can handle large datasets efficiently. 



---

## VII. License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## VIII. Contact

For any questions, suggestions, or contributions, please feel free to submit an issue or pull request on the GitHub repository. You can also reach out via [email](mailto:guillaumedt1001@gmail.com) or throuh my [LinkedIn](https://www.linkedin.com/in/guillaume-macquart-de-terline-a7b73430b) profile.
Thank you for your interest in this project! I hope it provides a valuable tool for density estimation in high-dimensional embedding spaces.