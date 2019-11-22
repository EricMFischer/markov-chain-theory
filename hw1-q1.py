import numpy as np
# import matplotlib.pyplot as plt

# K = np.matrix([[0.0, 0.7, 0.3, 0.0, 0.0], # test
#               [0.2, 0.0, 0.6, 0.0, 0.2],
#               [0.1, 0.4, 0.0, 0.5, 0.0],
#               [0.0, 0.3, 0.4, 0.0, 0.3],
#               [0.0, 0.0, 0.3, 0.7, 0]])
K = np.matrix([[0.3, 0.6, 0.1, 0.0, 0.0],
               [0.2, 0.0, 0.7, 0.0, 0.1],
               [0.0, 0.5, 0.0, 0.5, 0.0],
               [0.0, 0.0, 0.4, 0.1, 0.5],
               [0.4, 0.1, 0.0, 0.4, 0.1]])

# ------------------------------------------ PROBLEM 1 ------------------------------------------
# 1: calculate the 5 eigenvalues and their corresponding left and right eigenvectors
evals, evecs = np.linalg.eig(K.T)
print('evecs:', evecs)

# 1a: plot the 5 eigenvalues in a 2D plane, ie as dots in a unit circle
'''
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
t = np.linspace(0, np.pi*2, 100)
ax.plot(np.cos(t), np.sin(t), linewidth=1)
ax.scatter(evals.real, evals.imag, color='red')
fig.savefig('eigenvalues_unit_circle.png')
plt.show()
'''


# 1b: invariant probability pi
# left_evec = evecs[:, 0]
pi = evecs[:, 0] / np.linalg.norm(evecs[:, 0], ord=1)
left0 = np.matrix(pi).getH() # - or not?
print('left0:', left0.real)

# 1c: value of lambda_slem
descend_abs_evals = sorted(evals, key=abs, reverse=True)
lambda_slem = abs(descend_abs_evals[1].real)
print('evals:', descend_abs_evals)
print('λ_slem:', lambda_slem)



# 2: plot TV-norm and KL-divergence for 1000 steps
v = np.array([1,0,0,0,0]) # initial probabilities
x0 = 1 # initial state
steps = list(range(1, 1001))
stat_dist = np.linalg.matrix_power(K, 200)[0, :]
print('stat_dist:', stat_dist)

def tv_norm(dist, stat_dist):
  return np.sum(np.abs(stat_dist - dist)) / 2

def kl_divergence(dist, stat_dist):
  return np.sum(np.multiply(stat_dist, np.log(stat_dist / dist)))

mkv_chain_dist = [v*K**n for n in steps]
tv_norm_out = [tv_norm(mu, stat_dist) for mu in mkv_chain_dist]
kl_div = [kl_divergence(mu, stat_dist) for mu in mkv_chain_dist]
print('tv_norm_out:', tv_norm_out)
print('kl_div:', kl_div)

'''
fig = plt.figure()
plt.plot(steps[:100], tv_norm_out[:100], color='green', linewidth=1)
plt.plot(steps[:100], kl_div[:100], color='blue', linewidth=1)
plt.ylim(-0.1,1.1)
plt.grid()
plt.xlabel('n steps')
plt.ylabel('value')
plt.title('TV-norm and KL-divergence over n Steps')
plt.legend(['TV-norm', 'KL-divergence'])
fig.savefig('TV-norm-KL-divergence.png', dpi=300)
plt.show()
'''



# 3: calc. contraction coefficient C for K, ie max TV-norm between any 2 rows in transition kernel.
# if C < 1, then the convergence rate can be upper-bounded by A(n)
# plot the bound A(n) = C^n(K) for 1000 steps
def contraction_coeff(K):
  max_tv_norm = 0
  for i, row in enumerate(K):
    for j, row2 in enumerate(K[1:]):
      tv_norm_out = tv_norm(K[i, :], K[j, :])
      if tv_norm_out > max_tv_norm:
        max_tv_norm = tv_norm_out
  return max_tv_norm

C = contraction_coeff(K)
C_out = [C**n for n in steps]
print('C:', C)
print('C_out:', C_out)

'''
fig = plt.figure()
plt.plot(steps[:100], C_out[:100], color='red', linewidth=1)
plt.grid()
plt.ylim(-0.1,1.1)
plt.xlabel('n steps')
plt.ylabel('value')
plt.title('Bound A over n Steps')
plt.legend(['A(n) - Contraction Bound'])
fig.savefig('contraction_over_n_steps.png', dpi=300)
plt.show()
'''


# 4: another (diaconis-hanlon) bound we refer to as B(n)
# plot the real convergence rate of TV norm in comparison with A(n) and B(n)
# plot a second figure to compare their log plots as they are exponential rates
def diaconis_hanlon(stat_dist, lambda_slem, n, x0):
  stat_dist = stat_dist.reshape(5,1)
  frac = 1 - stat_dist[x0] / 4 * stat_dist[x0]
  return float(np.sqrt(frac) * lambda_slem**n)

dh_out = [diaconis_hanlon(stat_dist, lambda_slem, n, x0 - 1) for n in steps] # x0 - 1
print('dh_out:', dh_out)

'''
fig = plt.figure()
plt.plot(steps[:100], tv_norm_out[:100], color='green', linewidth=1)
plt.plot(steps[:100], C_out[:100], color='red', linewidth=1)
plt.plot(steps[:100], dh_out[:100], color='teal', linewidth=1)
plt.grid()
plt.xlabel('n steps')
plt.ylabel('value')
plt.title('TV-norm, Bound A, and Bound B over n Steps')
plt.legend(['TV-norm', 'A(n) - Contraction Bound', 'B(n) - DH Bound'])
fig.savefig('TV-norm-A-B-bounds.png', dpi=300)
plt.show()

fig = plt.figure()
plt.plot(steps[:100], tv_norm_out[:100], color='green', linewidth=1)
plt.plot(steps[:100], C_out[:100], color='red', linewidth=1)
plt.plot(steps[:100], dh_out[:100], color='teal', linewidth=1)
plt.grid()
plt.yscale('log')
plt.xlabel('n steps')
plt.ylabel('value')
plt.title('TV-norm, Bound A, and Bound B over n Steps')
plt.legend(['TV-norm', 'A(n) - Contraction Bound', 'B(n) - DH Bound'])
fig.savefig('TV-norm-A-B-bounds.png', dpi=300)
plt.show()
'''

# 5: define a new markov chain with transition kernel P = K^n, plot eigenvalues in 2D plane
# show how eigenvalues move on plane at 3 stages: n = 10, 100, 1000, tracing trajectories
# print the matrix P for n = 1000 to see if it becomes the "ideal" transition kernel

P_10 = np.linalg.matrix_power(K, 10)
P_100 = np.linalg.matrix_power(K, 100)
P_1000 = np.linalg.matrix_power(K, 1000)

evals_10, evecs_10 = np.linalg.eig(P_10.T)
evals_100, evecs_100 = np.linalg.eig(P_100.T)
evals_1000, evecs_1000 = np.linalg.eig(P_1000.T)
print('P_1000:', P_1000)


# 1a: plot the 5 eigenvalues in a 2D plane, ie as dots in a unit circle
'''
fig = plt.figure()
plt.grid()
ax = fig.add_subplot(1, 1, 1)
t = np.linspace(0, np.pi*2, 100)
ax.plot(np.cos(t), np.sin(t), linewidth=1)
ax.scatter(evals_10.real, evals_10.imag, color='red')
fig.savefig('eigenvalues_unit_circle_mult.png')
plt.show()

fig = plt.figure()
plt.grid()
ax = fig.add_subplot(1, 1, 1)
t = np.linspace(0, np.pi*2, 100)
ax.plot(np.cos(t), np.sin(t), linewidth=1)
ax.scatter(evals_100.real, evals_100.imag, color='blue')
fig.savefig('eigenvalues_unit_circle_mult.png')
plt.show()

fig = plt.figure()
plt.grid()
ax = fig.add_subplot(1, 1, 1)
t = np.linspace(0, np.pi*2, 100)
ax.plot(np.cos(t), np.sin(t), linewidth=1)
ax.scatter(evals_1000.real, evals_1000.imag, color='purple')
fig.savefig('eigenvalues_unit_circle_mult.png')
plt.show()
'''
