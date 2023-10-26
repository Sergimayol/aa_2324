import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

C_SIZE = 10_000

if __name__ == "__main__":
    # Generació del conjunt de mostres
    X, y = make_classification(
        n_samples=400,
        n_features=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1,
        random_state=9,
    )

    # Separar les dades: train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # Estandaritzar les dades: StandardScaler
    scaler = StandardScaler()
    X_train_transformed = scaler.fit_transform(X_train)

    # Entrenam una SVM linear (classe SVC)
    model = SVC(C=C_SIZE, kernel="linear")
    model.fit(X_train_transformed, y_train)

    # Prediccio
    X_test_transformed = scaler.transform(X_test)
    y_prediction = model.predict(X_test_transformed)

    # Metrica
    met = np.sum(y_prediction == y_test) / np.size(y_test) 
    print(met)
