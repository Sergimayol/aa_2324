from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

USE_MIN_MAX = True


def kernel_lineal(x1: np.ndarray, x2: np.ndarray) -> np.array:
    return x1.dot(x2.T)


def kernel_gauss(x1: np.ndarray, x2: np.ndarray, sigma: float = 1.0) -> np.array:
    return np.exp(distance_matrix(x1, x2) ** 2) * -sigma


def kernel_poly(
    x1: np.ndarray, x2: np.ndarray, d: int = 2, gamma: float = 10.0, r: float = 0.0
) -> np.array:
    return (gamma * x1.dot(x2.T) + r) ** d


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=0.5,
        random_state=8,
    )
    # En realitat ja no necessitem canviar les etiquetes Scikit ho fa per nosaltres
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # Els dos algorismes es beneficien d'estandaritzar les dades
    scaler = MinMaxScaler() if USE_MIN_MAX else StandardScaler()

    X_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    # LINEAR
    np_model = SVC(C=1.0, kernel="linear", random_state=33)
    my_model = SVC(C=1.0, kernel=kernel_lineal, random_state=33)

    np_model.fit(X_transformed, y_train)
    np_y = np_model.predict(X_test_transformed)

    my_model.fit(X_transformed, y_train)
    my_y = my_model.predict(X_test_transformed)

    print(
        "Son iguales? =",
        precision_score(np_y, my_y),
        precision_score(y_test, my_y),
        precision_score(y_test, np_y),
    )

    # GAUSSIAN
    np_model = SVC(C=1.0, kernel="rbf", random_state=33)
    my_model = SVC(C=1.0, kernel=kernel_gauss, random_state=33)

    np_model.fit(X_transformed, y_train)
    np_y = np_model.predict(X_test_transformed)

    my_model.fit(X_transformed, y_train)
    my_y = my_model.predict(X_test_transformed)

    print(
        "Son iguales? =",
        precision_score(np_y, my_y),
        precision_score(y_test, my_y),
        precision_score(y_test, np_y),
    )

    # POLYNOMIAL
    np_model = SVC(C=1.0, kernel="poly", random_state=33)
    my_model = SVC(C=1.0, kernel=kernel_poly, random_state=33)

    np_model.fit(X_transformed, y_train)
    np_y = np_model.predict(X_test_transformed)

    my_model.fit(X_transformed, y_train)
    my_y = my_model.predict(X_test_transformed)

    print(
        "Son iguales? =",
        precision_score(np_y, my_y),
        precision_score(y_test, my_y),
        precision_score(y_test, np_y),
    )
