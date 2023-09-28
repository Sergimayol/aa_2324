import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

SGD = False
# Com les classes es diuen igual basta canviar el fitxer ;)
if SGD:
    from AdaLine_SGD import Adaline
else:
    from AdaLine_Batch import Adaline

# Generació del conjunt de mostres
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=2,
    random_state=9,
)

y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.

perceptron = Adaline(eta=1e-4, n_iter=300)
scaler = StandardScaler()
X = scaler.fit_transform(X)
perceptron.fit(X, y)

y_prediction = perceptron.predict(X)

#  Mostram els resultats
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)

# Dibuixem la recta de separació
m = -perceptron.w_[1] / perceptron.w_[2]
origen = (0, -perceptron.w_[0] / perceptron.w_[2])
plt.axline(xy1=origen, slope=m)

plt.figure(2)
plt.plot(perceptron.cost_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Sum of squared error")
plt.show()
