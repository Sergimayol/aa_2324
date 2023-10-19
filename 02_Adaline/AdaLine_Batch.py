from tqdm import trange
import numpy as np


class Adaline:
    """ADAptive LInear NEuron classifier.
       Gradient Descent

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Error in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.cost_ = []
        self.w_ = np.zeros(1)  # Avoid warnings

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []  # Per calcular el cost a cada iteració (EXTRA)

        for i in (t := trange(self.n_iter)):
            err = y - self.predict(X)
            self.w_[1:] += np.dot(X.T, err) * self.eta
            self.w_[0] += np.sum(err) * self.eta
            loss = np.sum(err**2) * 0.5
            self.cost_.append(loss)
            t.set_description(f"Epoch: {i+1}, Loss: {loss}")

    def predict(self, X):
        """Return class label after unit step"""
        res = self.w_.T
        return np.dot(X, res[1:]) + res[0]
