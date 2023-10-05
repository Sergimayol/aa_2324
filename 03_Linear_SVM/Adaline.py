import numpy as np
from tqdm import trange

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

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

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

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
