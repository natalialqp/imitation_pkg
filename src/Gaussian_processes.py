import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import GaussianSymmetrizedKLKernel, ScaleKernel
from gpytorch.means import ConstantMean
import os

def plot_gpr_samples(gpr_model, n_samples, ax):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(0, 1000, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
    ax.plot(x, y_mean, color="black", label="Mean")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-3, 3])

def gpr_rbf_kernel(X_train, y_train, n_samples):
    # X_train = X_train.reshape(-1, 1)
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(15, 10))
    # plot prior
    plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
    axs[0].set_title("Samples from prior distribution")

    # plot posterior
    gpr.fit(X_train, y_train)
    plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
    axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
    axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
    axs[1].set_title("Samples from posterior distribution")

    fig.suptitle("Radial Basis Function kernel", fontsize=18)
    plt.tight_layout()
    plt.show()

def gpr_sklearn(y, X):
    time = X
    X = X.reshape(-1, 1)
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size=15, replace=False)
    X_train, y_train = X[training_indices], y[training_indices]
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_train)

    gaussian_process.kernel_
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
    print(mean_prediction)
    plt.plot(X, y, label=r"$f(x)$", linestyle="dotted")
    plt.scatter(X_train, y_train, label="Observations")
    plt.plot(time, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title("Gaussian process regression on noise-free dataset")
    plt.show()

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(GaussianSymmetrizedKLKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def gpr_pytorch(train_y):
    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x_mean = torch.linspace(0, 1, len(train_y))
    # We'll assume the variance shrinks the closer we get to 1
    train_x_stdv = torch.linspace(0.03, 0.01, len(train_y))
    # True function is sin(2*pi*x) with Gaussian noise
    # train_y = torch.sin(train_x_mean * (2 * math.pi)) + torch.randn(train_x_mean.size()) * 0.2
    train_y = torch.Tensor(train_y)
    train_x_distributional = torch.stack((train_x_mean, (train_x_stdv**2).log()), dim=1)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x_distributional, train_y, likelihood)

    smoke_test = ('CI' in os.environ)
    training_iter = 2 if smoke_test else 30

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.25)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x_distributional)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        test_x_distributional = torch.stack((test_x, (1e-3 * torch.ones_like(test_x)).log()), dim=1)
        observed_pred = likelihood(model(test_x_distributional))

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(8, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.errorbar(train_x_mean.numpy(), train_y.numpy(), fmt='k')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()


if __name__ == "__main__":
    # X = np.linspace(start=0, stop=10, num=1_000)
    # y = np.squeeze(X * np.sin(X))
    # gpr_sklearn(y, X)

    # gpr_pytorch(y)

    rng = np.random.RandomState(4)
    X_train = rng.uniform(0, 1000, 10000).reshape(-1, 1)
    y_train = np.sin((X_train[:, 0] - 2.5) ** 2)
    n_samples = 15
    gpr_rbf_kernel(X_train, y_train, n_samples)