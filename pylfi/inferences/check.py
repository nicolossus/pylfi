import numpy as np
import pathos as pa
import pylfi
import scipy.stats as stats

# prior distribution
prior = pylfi.Prior('norm', loc=0, scale=1, name='theta')
prior_pdf = prior.pdf
prior_logpdf = prior.logpdf

# draw from prior distribution
#thetas_current = [prior.rvs()]
thetas_current = [prior.rvs(), prior.rvs()]

# proposal distribution
sigma = 0.5
proposal_distr = stats.norm(loc=thetas_current, scale=sigma)
uniform_distr = stats.uniform(loc=0, scale=1)

# draw from proposal
thetas_proposal = [proposal_distr.rvs()]
print(thetas_proposal)
for theta in thetas_proposal:
    print(theta)

print()
# Compute prior probability of current and proposed
prior_current = prior_pdf(thetas_current)
prior_proposal = prior_pdf(thetas_proposal)
log_prior_current = prior_logpdf(thetas_current).prod()
log_prior_proposal = prior_logpdf(thetas_proposal).prod()

# since the proposal density is symmetric, the proposal density ratio in MH
# acceptance probability cancel. Thus, we need only to evaluate the prior ratio

# no need to evaluate the MH-ratio, but check that prior > 0

r = np.exp(log_prior_proposal - log_prior_current)
alpha = np.minimum(1., r)
u = uniform_distr.rvs()
print(r)
print(alpha)
print(u)
print(u < alpha)

prior = pylfi.Prior('norm', loc=0, scale=1, name='theta')
prior2 = pylfi.Prior('norm', loc=0, scale=1, name='theta2')

priors = [prior, prior2]

prior_logpdfs = [prior.logpdf for prior in priors]

log_prior_current = np.array([prior_logpdf(theta_current)
                              for prior_logpdf, theta_current in
                              zip(prior_logpdfs, thetas_current)]
                             ).prod()

log_prior_proposal = np.array([prior_logpdf(thetas_proposal)
                               for prior_logpdf, thetas_proposal in
                               zip(prior_logpdfs, thetas_proposal)]
                              ).prod()

print(" ")
# print(log_prior_current)
# print(log_prior_proposal)
r = np.exp(log_prior_proposal - log_prior_current)
alpha = np.minimum(1., r)
print(r)
print(alpha)
print(u < alpha)


'''
print(f"{thetas_current=}")
print(f"{thetas_proposal=}")
print(f"{prior_current=}")
print(f"{prior_proposal=}")
print(f"{log_prior_current=}")
print(f"{log_prior_proposal=}")
'''

'''

def proposal(a_c, b_c):
    a_prop = np.random.normal(loc=a_c,scale=0.1)
    b_prop = np.random.normal(loc=b_c,scale=0.1)
    return a_prop, b_prop

def alpha(a_prop, b_prop, a_i, b_i):
    n1 = stats.uniform(0.1,1.5).pdf(a_prop)*stats.uniform(0.1,1.5).pdf(b_prop)
    n2 = stats.norm(a_prop,0.2).pdf(a_i)*stats.norm(b_prop,0.2).pdf(b_i)
    d1 = stats.uniform(0.1,1.5).pdf(a_i)*stats.uniform(0.1,1.5).pdf(b_i)
    d2 = stats.norm(a_i,0.2).pdf(a_prop)*stats.norm(b_i,0.2).pdf(b_prop)
    return min(1, (n1*n2)/(d1*d2))
'''


'''
# Compute likelihood by multiplying probabilities of each data point
likelihood_current = stats.norm(mu_current + 20, 1).logpdf(data).sum()
likelihood_proposal = stats.norm(
    mu_proposal + 20, 1).logpdf(data).sum()

# Compute prior probability of current and proposed mu
prior_current = prior_logpdf(mu_current)
prior_proposal = prior_logpdf(mu_proposal)

# log(p(x|θ) p(θ)) = log(p(x|θ)) + log(p(θ))
p_current = likelihood_current + prior_current
p_proposal = likelihood_proposal + prior_proposal

# Accept proposal?
p_accept = np.exp(p_proposal - p_current)
accept = np.random.rand() < p_accept

# draw proposal parameters
# thetas_proposal = [proposal_distr.rvs()
#                   for _ in range(len(self._priors))]
'''
