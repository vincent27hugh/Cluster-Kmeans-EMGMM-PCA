In this project, we cluster different types of wines using use [Wine Dataset](http://archive.ics.uci.edu/ml/datasets/Wine) and cluster algorithms such [K-Means](https://en.wikipedia.org/wiki/K-means_clustering), [Expectation Maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) - [Gaussian Mixture Model](https://en.wikipedia.org/wiki/Mixture_model) (EM-GMM), and [Principle Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA).

There are 13 features (1. Alcohol 2. Malicacid 3. Ash 4. Alcalinity of ash 5. Magnesium 6. Total phenols 7. Flavanoids 8. Nonflavanoid phenol) for us to feed into the cluster algorithms. We can take use of them directly or commit dimension reduction at first. Thus we 

* K-Means & EM-GMM:
	* apply K-Means and EM-GMM using 13 features directly and compare the performance;
* PCA & K-Means & EM-GMM:
	* apply PCA to select top 6 PCs;
	* apply K-Means and EM-GMM using the selected 6 Pcs and compare the performance;

We also visualize the cluster result using only top 2 PCs.