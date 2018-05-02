import numpy as np
import matplotlib.pyplot as plt

def gdl1(X, y, a, lamb1=0, lamb2=0):
	result = np.dot(X, a)
	return 2 * np.dot(X.T, y - result) - lamb1

trials = 1000
steps = 1000000
lamb1 = 0.001
lr = 0.00005

train_n = 100
test_n = 10000
d = 200

results = []

for _ in range(trials):
	X_train = np.random.normal(0,1, size=(train_n,d))
	a_true = np.random.normal(0,1, size=(d,1))
	y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
	X_test = np.random.normal(0,1, size=(test_n,d))
	y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))


	a = np.zeros((d, 1))
	order = np.random.permutation(train_n)
	for e in range(steps//train_n):
		for s in range(train_n):
			a += lr * gdl1(X_train[order[s]].reshape(1, d), y_train[order[s]], a, lamb1)
	results.append(np.linalg.norm(X_test.dot(a) - y_test) / np.linalg.norm(y_test))

print(np.mean(results))