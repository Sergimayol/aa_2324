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

    def __init__(self, eta=0.01, n_iter=10, verbose: bool = False):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.__v = verbose
        self.errors_ = None  # defined in method fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        """
        accuracy = []
        self.errors_ = []
        # First position corresponds to threshold
        self.w_ = np.zeros(1 + X.shape[1])
        for i in (t := tqdm(range(self.n_iter), position=0)):
            errors = 0
            for xi, target in zip(X, y):
                # Calculate the output: (X * weights) + threshold
                output = self.predict(xi)
                # Update weights
                update = self.eta * (target - output)
                self.w_[1:] += update * xi
                self.w_[0] += update
                # Count miss classifications
                if self.__v:
                    errors += int(update != 0.0)

            if self.__v:
                # Calculate accuracy
                accuracy.append(self.__acurracy(self.predict(X), y))
                self.errors_.append(errors)
            acc = f", acc: {accuracy[i]}" if self.__v else ""
            t.set_description(f"Epoch {i + 1}{acc}")

        return accuracy, self.errors_

    def __acurracy(
        self, y_prediction: np.ndarray, y: np.ndarray, round: int = 4
    ) -> float:
        """Calculate the acurracy."""
        return np.round(np.sum(y_prediction == y) / len(y), round)

    def __net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label.
        First calculate the output: (X * weights) + threshold
        Second apply the step function
        Return a list with classes
        """
        return np.where(self.__net_input(X) >= 0.0, 1, -1)
