import numpy as np

K = np.matrix([[0.0, 0.7, 0.3, 0.0, 0.0], # test
               [0.2, 0.0, 0.6, 0.0, 0.2],
               [0.1, 0.4, 0.0, 0.5, 0.0],
               [0.0, 0.3, 0.4, 0.0, 0.3],
               [0.0, 0.0, 0.3, 0.7, 0]])
K1 = np.matrix([[0.1, 0.4, 0.3, 0.0, 0.2],
                [0.5, 0.3, 0.2, 0.0, 0.0],
                [0.0, 0.4, 0.5, 0.1, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.7, 0.3]])
K2 = np.matrix([[0.0, 0.0, 0.0, 0.4, 0.6],
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.9, 0.1],
                [0.0, 0.2, 0.8, 0.0, 0.0],
                [0.3, 0.0, 0.7, 0.0, 0.0]])

# ------------------------------------------ PROBLEM 2 ------------------------------------------
# 2b: calculate the 5 eigenvalues and 5 eigenvectors of the two matrices
K_evals, K_evecs = np.linalg.eig(K.T)
K1_evals, K1_evecs = np.linalg.eig(K1.T)
K2_evals, K2_evecs = np.linalg.eig(K2.T)
print('K evals:', K_evals)
print('K evecs:', K_evecs)
print('K1 evals:', K1_evals)
print('K1 evecs:', K1_evecs)
print('K2 evals:', K2_evals)
print('K2 evecs:', K2_evecs)

# 2c: how many and what are the invariant probabilities for each matrix?
pi_K = K_evecs[:, 0] / np.linalg.norm(K_evecs[:, 0], ord=1)
left0_K = np.matrix(pi_K).getH()
print('left0_K:', left0_K.real)

pi_K1 = K1_evecs[:, 0] / np.linalg.norm(K1_evecs[:, 0], ord=1)
left0_K1 = np.matrix(pi_K1).getH()
print('left0_K1:', left0_K1.real)

pi_K2 = K2_evecs[:, 0] / np.linalg.norm(K2_evecs[:, 0], ord=1)
left0_K2 = np.matrix(pi_K2).getH()
print('left0_K2:', left0_K2.real)
