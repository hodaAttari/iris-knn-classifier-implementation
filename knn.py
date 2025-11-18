import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MyKNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple K-Nearest Neighbors classifier.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors (K).
    metric : str
        Distance metric to use. Supported: "euclidean", "manhattan".
    weighted : bool
        If True, weight neighbors by distance.
    """
    def __init__(self, n_neighbors=3, metric="euclidean", weighted=False):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weighted = weighted

        # these will be set in fit()
        self.X_train = None
        self.y_train = None
        
    def _fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _compute_distances(self, X):
        if self.metric == "euclidean":
            return np.sqrt(((X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2).sum(axis=2))
        elif self.metric == "manhattan":
            return np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]).sum(axis=2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        
    def _predict_one(self, distances_row):
        idx = np.argsort(distances_row)[:self.n_neighbors]
        neighbor_labels = self.y_train[idx]
        if self.weighted:
            neighbor_dist = distances_row[idx]
            weights = 1 / (neighbor_dist + 1e-9)
            unique_labels = np.unique(neighbor_labels)
            weighted_votes = [weights[neighbor_labels == lbl].sum() for lbl in unique_labels]
            return unique_labels[np.argmax(weighted_votes)]
        else:
            # رأی اکثریت ساده
            values, counts = np.unique(neighbor_labels, return_counts=True)
            return values[np.argmax(counts)]
        
        
    def _predict(self, X):
        X = np.array(X)
        dists = self._compute_distances(X)
        return np.array([self._predict_one(row) for row in dists])

    # def fit(self, X, y):
    #     """
    #     Store the training data.
    #     KNN is a lazy learner, so no actual training happens here.
    #     """
    #     # TODO: store X and y inside the object
    #     # Example:
    #     # self.X_train = ...
    #     # self.y_train = ...
    #     #
    #     # return self (very important for sklearn compatibility)
    #     raise NotImplementedError

    # 
    
    # def predict(self, X):
    #     """
    #     Predict labels for multiple samples in X.

    #     Returns
    #     -------
    #     y_pred : np.ndarray of shape (num_test,)
    #     """
    #     # TODO:
    #     # 1. compute full distance matrix between X and self.X_train
    #     # 2. for each row of that matrix, call _predict_one(...)
    #     raise NotImplementedError

