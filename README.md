# ornstein-uhlenbeck

Estimate the parameters of an Ornstein-Uhlenbeck stochastic process (also known
as a Vasicek model) using maximum likelihood for eta (the attraction parameter)
and iterative updating for mu and sigma.

The stochastic equation is:
dX = - eta * (X - mu) * dt + sigma * dW
where W is a Wigner process.

## Usage

The estimator object takes a list of pairs of arrays so that multiple
independent datasets can be evaluated at once. The number of iterations can be
specified with the `n_it` keyword argument. For many large datasets, a high
number of iterations is not needed and the default is only one.

``` python
import numpy as np
import ornstein_uhlenbeck as ou

t_data = np.array([1, 2, 4, 5])
x_data = np.array([0.2, 1.2, 0.4, -0.3])
estimator = ou.OrnsteinUhlenbeckEstimator([(t_data, x_data)], n_it=3)

print(f'mu = {estimator.mu}')
print(f'eta = {estimator.eta}')
print(f'sigma^2 = {estimator.sigma_sq()}')
```

