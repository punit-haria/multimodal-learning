import numpy as np

# MADE network masking example


n_a = 6
n_b = 10
n_c = 10
n_d = 6

m_prev = np.arange(n_a) + 1
m = np.random.randint(low=1, high=n_a, size=n_b)

mask1 = np.zeros(shape=[n_a, n_b], dtype=np.float32)
for a in range(n_a):
    for b in range(n_b):
        if m[b] >= m_prev[a]:
            mask1[a, b] = 1

m_prev = m
min_val = np.min(m_prev)
m = np.random.randint(low=min_val, high=n_b, size=n_c)

mask2 = np.zeros(shape=[n_b, n_c], dtype=np.float32)
for b in range(n_b):
    for c in range(n_c):
        if m[c] >= m_prev[b]:
            mask2[b, c] = 1

m_prev = m
m = np.arange(n_d)

mask3 = np.zeros(shape=[n_c, n_d], dtype=np.float32)
for c in range(n_c):
    for d in range(n_d):
        if m[d] >= m_prev[c]:
            mask3[c, d] = 1


temp = np.matmul(mask1, mask2)
np.matmul(temp, mask3)
























