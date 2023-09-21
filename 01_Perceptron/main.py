import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from Perceptron import Perceptron


def regression_line(w):
    max_x = np.max(X) + 1
    min_x = np.min(X) - 1
    slope = -(w[0] / w[2]) / (w[0] / w[1])
    intercept = -w[0] / w[2]
    x_plot = np.linspace(min_x, max_x)
    y_plot = slope * x_plot + intercept
    return x_plot, y_plot


if __name__ == "__main__":
    # Generació del conjunt de mostres
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.25,
        random_state=0,
    )

    y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.

    perceptron = Perceptron(n_iter=150)
    perceptron.fit(X, y)  # Ajusta els pesos
    y_prediction = perceptron.predict(X)  # Prediu

    x_plot, y_plot = regression_line(perceptron.w_)

    #  Resultats
    plt.figure(1)
    plt.scatter(
        X[:, 0], X[:, 1], c=y_prediction
    )  # Mostram el conjunt de mostres el color indica la classe
    plt.plot(x_plot, y_plot, color="black")  # Mostram la recta de regressió
    plt.title("Perceptron")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
