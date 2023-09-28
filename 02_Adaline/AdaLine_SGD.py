from tqdm import tqdm
import numpy as np
from numpy.random import seed


class Adaline(object):
    """ADAptive LInear NEuron classifier.

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
        Number of misclassifications in every epoch.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.

    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.cost_ = []
        self.w_ = np.zeros(1) # Avoid warnings

        if random_state:
            seed(random_state)

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

        for i in (t := tqdm(range(self.n_iter))):
            if self.shuffle:
                X, y = self.__shuffle(X, y)
            errs = []
            for xi, target in zip(X, y):
                output = self.predict(xi)
                err = target - output
                errs.append(err**2)
                update = self.eta * err
                self.w_[1:] += update * xi
                self.w_[0] += update

            cost = np.sum(errs) * 0.5
            self.cost_.append(cost)
            t.set_description(f"Epoch: {i+1}")

    def __shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def predict(self, X):
        """Return class label after unit step"""
        res = self.w_.T
        return np.dot(X, res[1:]) + res[0]
