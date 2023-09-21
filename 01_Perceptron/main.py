import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from Perceptron import Perceptron


def regression_line(w: np.ndarray[np.float64], max_x: float = 1, min_x: float = -1):
    apox = lambda x: x if x != 0 else 0.00001
    w[0] = apox(w[0])
    w[1] = apox(w[1])
    w[2] = apox(w[2])
    slope = -(w[0] / w[2]) / (w[0] / w[1])
    intercept = -w[0] / w[2]
    x_plot = np.linspace(min_x, max_x)
    y_plot = slope * x_plot + intercept
    return x_plot, y_plot


def acurracy(y_prediction: np.ndarray, y: np.ndarray, round: int = 4) -> float:
    return np.round(np.sum(y_prediction == y) / len(y), round)


if __name__ == "__main__":
    # Generació del conjunt de mostres
    X, y = make_classification(
        n_samples=10000,
        n_features=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.75,
        random_state=0,
    )

    y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.

    perceptron = Perceptron(n_iter=200)  # Creació del perceptron
    perceptron.fit(X, y)  # Ajusta els pesos
    y_prediction = perceptron.predict(X)  # Prediu

    x_plot, y_plot = regression_line(perceptron.w_, np.max(X[:, 0]), np.min(X[:, 0]))

    #  Resultats
    plt.figure(1)
    plt.scatter(
        X[:, 0], X[:, 1], c=y_prediction
    )  # Mostram el conjunt de mostres el color indica la classe
    plt.plot(x_plot, y_plot, color="black")  # Mostram la recta de regressió
    plt.title(f"Perceptron - Acurracy: {acurracy(y_prediction, y)}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
