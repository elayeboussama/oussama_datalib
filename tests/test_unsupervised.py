import pytest
from src.datalib.unsupervised import perform_kmeans, perform_pca, perform_gaussian_mixture
import numpy as np

def test_perform_kmeans():
    """Test performing KMeans clustering."""
    X = np.random.rand(100, 2)
    model = perform_kmeans(X, n_clusters=3)
    assert model.n_clusters == 3

def test_perform_pca():
    """Test performing PCA."""
    X = np.random.rand(100, 5)
    X_reduced, explained_variance, pca = perform_pca(X, n_components=2)
    assert X_reduced.shape[1] == 2
    assert len(explained_variance) == 2

def test_perform_gaussian_mixture():
    """Test performing Gaussian Mixture modeling."""
    X = np.random.rand(100, 2)
    model = perform_gaussian_mixture(X, n_components=3)
    assert model.n_components == 3
