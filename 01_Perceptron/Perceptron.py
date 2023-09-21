import numpy as np
from tqdm import tqdm


class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of miss classifications in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None  # defined in method fit

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
        # First position corresponds to threshold
        self.w_ = np.zeros(1 + X.shape[1])
        for i in (t := tqdm(range(self.n_iter), leave=True, position=0)):
            t.set_description(f"Epoch {i + 1}", refresh=True)
            for xi, target in zip(X, y):
                # Calculate the output: (X * weights) + threshold
                output = self.predict(xi)
                # Update weights
                update = self.eta * (target - output)
                self.w_[1:] += update * xi
                self.w_[0] += update

    def predict(self, X):
        """Return class label.
        First calculate the output: (X * weights) + threshold
        Second apply the step function
        Return a list with classes
        """
        res = np.dot(X, self.w_[1:]) + self.w_[0]
        return np.where(res >= 0.0, 1, -1)
