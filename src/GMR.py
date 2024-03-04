"""
=====================================
Linear Gaussian Models for Regression
=====================================

In this example, we use a MVN to approximate a linear function and a mixture
of MVNs to approximate a nonlinear function. We estimate p(x, y) first and
then we compute the conditional distribution p(y | x).
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from gmr.utils import check_random_state
from gmr import MVN, GMM, plot_error_ellipses


def test_gmr(data, time):
    X = np.ndarray((len(time), 2))
    X[:, 0] = time
    X[:, 1] = data
    n_samples = len(time) + 100
    n_samples = 1000
    X_test = np.linspace(0, n_samples, n_samples)

    plt.figure(figsize=(10, 5))

    gmm = GMM(n_components=3, random_state=0)
    gmm.from_samples(X)
    Y = gmm.predict(np.array([0]), X_test[:, np.newaxis])

    plt.title("Gaussian Mixture Regression (GMR)")

    plt.scatter(X[:, 0], X[:, 1])
    plot_error_ellipses(plt.gca(), gmm, colors=["r", "g", "b"])
    plt.plot(X_test, Y.ravel(), c="k", lw=2, label="GMR")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    plt.show()

if __name__ == "__main__":
    n_samples = 100
    X = np.ndarray((n_samples, 2))
    X[:, 0] = np.linspace(0, 2 * np.pi, n_samples)
    random_state = check_random_state(0)
    X[:, 1] = np.sin(X[:, 0]) + random_state.randn(n_samples) * 0.1

    test_gmr(X[:, 1], X[:, 0])