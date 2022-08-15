import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import plot_tree


iris = load_breast_cancer()
# Parameters
n_classes = 2
plot_colors = ["C0","C1"]
plot_step = 0.02


for pairidx, pair in enumerate([[0, 1]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=1, min_samples_split=2).fit(X, y)

    # Plot the decision boundary
    ax = plt.subplot(1,1,pairidx + 1)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel="feature: " + iris.feature_names[pair[0]],
        ylabel="feature: " + iris.feature_names[pair[1]],
    )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

plt.suptitle("Borstkanker: beslisoppervlak van beslissingsboom met stopcriteria")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")


plt.figure()
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=1, min_samples_split=2).fit(X, y)
plot_tree(clf, filled=True)
plt.title("Borstkanker: beslissingsboom met stopcriteria ")
plt.show()