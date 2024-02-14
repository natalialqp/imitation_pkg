import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn import mixture
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline
import os
import pandas as pd

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

class GaussianMixturePlotter:
    def __init__(self, robotName, actionName, angleId, n_components=10, covariance_type="full", max_iter=150, random_state=2):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.robotName = robotName
        self.actionName = actionName
        self.angleId = angleId

    def fit_gaussian_mixture(self, X):
        gmm = mixture.GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state
        ).fit(X)
        return gmm

    def fit_bayesian_gaussian_mixture(self, X, weight_concentration_prior=1e-2,
                                      weight_concentration_prior_type="dirichlet_process",
                                      mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
                                      init_params="random"):
        dpgmm = mixture.BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            weight_concentration_prior=weight_concentration_prior,
            weight_concentration_prior_type=weight_concentration_prior_type,
            mean_precision_prior=mean_precision_prior,
            covariance_prior=covariance_prior,
            init_params=init_params,
            max_iter=self.max_iter,
            random_state=self.random_state
        ).fit(X)
        return dpgmm

    def smooth_curve(self, x, y, window_size=10):
        # Apply a simple moving average to smooth the curve
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        x_smooth = x[:len(y_smooth)]
        return x_smooth, y_smooth

    def plot_results(self, X, Y, avg_signal, means, covariances, index, title):
        splot = plt.subplot(2, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            if not np.any(Y == i):
                continue
            plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color, label=f'Component {i + 1}')
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi
            ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        initial = [X[0][0], avg_signal[0]]
        last = [X[-1][0], avg_signal[-1]]

        # Compute cluster centroids
        centroids = np.array([np.nanmean(X[Y == i], axis=0) for i in range(self.n_components)])
        centroids = np.vstack([initial, centroids, last])
        # Remove NaN values if any
        valid_mask = ~np.isnan(centroids[:, 0])
        centroids = centroids[valid_mask]
        if np.isnan(centroids).any():
            centroids = self.impute_nan_values(centroids)

        # Sort centroids based on x-values
        # centroids = np.vstack([X[Y == 0][0], centroids])
        sorted_centroids = centroids[np.argsort(centroids[:, 0])]
        # Plot smooth curve using make_interp_spline for all centroids
        spl = make_interp_spline(sorted_centroids[:, 0], sorted_centroids[:, 1], k=2)
        x_smooth_range = np.linspace(sorted_centroids[:, 0].min(), sorted_centroids[:, 0].max(), 100)
        y_smooth_range = spl(x_smooth_range)

        file_path = 'GMM_learned_actions/' + self.robotName + '_' + self.actionName + '.csv'
        if not os.path.isfile(file_path):
            header = 'time'
            np.savetxt(file_path, x_smooth_range, delimiter=',', header=header, comments='')
            existing_header = [header]
        else:
            existing_header = pd.read_csv(file_path, index_col=0).columns.tolist()
            existing_header.insert(0, 'time')
        existing_data = np.genfromtxt(file_path, delimiter=',',  skip_header=1)  # Skip the header row
        # Check if self.angleId exists in the header
        if self.angleId in existing_header:
            # If self.angleId exists, overwrite the corresponding column
            angle_id_index = np.where(np.array(existing_header) == self.angleId)[0][0]
            existing_data[:, angle_id_index] = y_smooth_range
        else:
            # If self.angleId does not exist, create the header and append the new column
            header_list = list(existing_header)
            header_list.append(self.angleId)
            header = ','.join(header_list)
            existing_data = np.column_stack((existing_data, y_smooth_range))
            np.savetxt(file_path, existing_data, delimiter=',', header=header, comments='')
        # Apply moving average to smooth the curve
        plt.plot(x_smooth_range, y_smooth_range, color='red', linestyle='-', linewidth=2, label='Smoothed Line Across Clusters')

        plt.title(title)
        plt.grid(True)
        plt.legend()

    def plot_samples(self, X, Y, n_components, index, title):
        plt.subplot(5, 1, 4 + index)
        for i, color in zip(range(n_components), color_iter):
            if not np.any(Y == i):
                continue
            plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color, label=f'Component {i + 1}')

        plt.title(title)
        plt.grid(True)
        plt.ylabel('Value')
        plt.xlabel('Time')

    def impute_nan_values(self, data):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        return imputer.fit_transform(data)

def eval(X, avg_signal, robotName, actionName, angleId, n_components):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(
        bottom=0.04, top=0.95, hspace=0.2, wspace=0.05, left=0.03, right=0.97
    )
    plotter = GaussianMixturePlotter(robotName, actionName, angleId, n_components)

    # Fit a Gaussian mixture with EM using ten components
    gmm = plotter.fit_gaussian_mixture(X)

    plotter.plot_results(X, gmm.predict(X), avg_signal, gmm.means_, gmm.covariances_, 0, "Expectation-maximization")

    # Fit Bayesian Gaussian mixture models with a Dirichlet process prior
    # dpgmm = plotter.fit_bayesian_gaussian_mixture(X)
    # plotter.plot_results(
        # X,
        # dpgmm.predict(X),
        # dpgmm.means_,
        # dpgmm.covariances_,
        # 1,
        # "Bayesian Gaussian mixture models with a Dirichlet process prior "
        # r"for $\gamma_0=0.01$.",
    # )

    # Fit Bayesian Gaussian mixture models with a Dirichlet process prior for $\gamma_0=100$
    dpgmm = plotter.fit_bayesian_gaussian_mixture(X, weight_concentration_prior=1e2, init_params="kmeans")
    plotter.plot_results(
        X,
        dpgmm.predict(X),
        avg_signal,
        dpgmm.means_,
        dpgmm.covariances_,
        1,
        "Bayesian Gaussian mixture models with a Dirichlet process prior "
        r"for $\gamma_0=100$",
    )
    plt.legend()
    plt.show()

if __name__ == "__main__":
    n_samples = 100
    np.random.seed(0)
    X = np.zeros((n_samples, 2))
    step = 4.0 * np.pi / n_samples
    for i in range(X.shape[0]):
        x = i * step
        X[i, 0] = x + np.random.normal(0, 0.1)
        X[i, 1] = 3.0 * (np.sin(x) + np.random.normal(0, 0.2))
    eval(X, n_components=5)
