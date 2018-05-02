import numpy as np
import matplotlib.pyplot as plt

def gdl2(X, y, a):
    result = np.dot(X, a)
    return 2 * np.dot(X.T, y - result)

d = 100 # dimensions of data
n = 1000 # number of data points
X = np.random.normal(0,1, size=(n,d))
a_true = np.random.normal(0,1, size=(d,1))
y = X.dot(a_true) + np.random.normal(0,0.5,size=(n,1))

#part a
a = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print("(a)", np.linalg.norm(X.dot(a)-y) ** 2)
print("(a_zeros)", np.linalg.norm(y) ** 2)

#part b
bins = np.zeros((20, 3))
b_1 = np.zeros((d,1))
b_2 = np.zeros((d,1))
b_3 = np.zeros((d,1))

for i in range(20):
    b_1 += 0.00005 * gdl2(X, y, b_1)
    b_2 += 0.0005 * gdl2(X, y, b_2)
    b_3 += 0.0007 * gdl2(X, y, b_3)

    bins[i, 0] = np.linalg.norm(X.dot(b_1) - y) ** 2
    bins[i, 1] = np.linalg.norm(X.dot(b_2) - y) ** 2
    bins[i, 2] = np.linalg.norm(X.dot(b_3) - y) ** 2

plt.plot(bins[:, 0], label='0.00005')
plt.plot(bins[:, 1], label='0.0005')
plt.plot(bins[:, 2], label='0.0007')
plt.legend()
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Train Loss')
plt.savefig('1b.png', bbox_inches='tight')
plt.clf()

#part c     
a0 = np.zeros((d, 1))
a1 = np.zeros((d, 1))
a2 = np.zeros((d, 1))

bins3 = np.zeros((1000, 3))
steps = [0.0005, 0.005, 0.01]
order = np.random.permutation(n)
for s in range(n):
    a0 += steps[0] * gdl2(X[order[s]].reshape(1, d), y[order[s]], a0)
    a1 += steps[1] * gdl2(X[order[s]].reshape(1, d), y[order[s]], a1)
    a2 += steps[2] * gdl2(X[order[s]].reshape(1, d), y[order[s]], a2)

    bins3[s, 0] = np.linalg.norm(X.dot(a0)-y) ** 2
    bins3[s, 1] = np.linalg.norm(X.dot(a1)-y) ** 2
    bins3[s, 2] = np.linalg.norm(X.dot(a2)-y) ** 2

plt.plot(bins3[:, 0], label='0.0005')
plt.plot(bins3[:, 1], label='0.005')
plt.plot(bins3[:, 2], label='0.01')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Train Loss')
plt.savefig('1c.png', bbox_inches='tight')