

def OLS(theta, simulation, observation):
    """
    theta - accepted thetas
    simulation - accepted simulated summary statistics 
    observation - observation summary statistic
    """
    return theta_adjusted


def run_lra(
    theta: torch.Tensor,
    x: torch.Tensor,
    observation: torch.Tensor,
    sample_weight=None,
) -> torch.Tensor:
    """Return parameters adjusted with linear regression adjustment.
    Implementation as in Beaumont et al. 2002: https://arxiv.org/abs/1707.01254
    """

    theta_adjusted = theta
    for parameter_idx in range(theta.shape[1]):
        regression_model = LinearRegression(fit_intercept=True)
        regression_model.fit(
            X=x,
            y=theta[:, parameter_idx],
            sample_weight=sample_weight,
        )
        theta_adjusted[:, parameter_idx] += regression_model.predict(
            observation.reshape(1, -1)
        )
        theta_adjusted[:, parameter_idx] -= regression_model.predict(x)

    return theta_adjusted
