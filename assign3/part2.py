import numpy as np
import matplotlib.pyplot as plt

train_n = 100
test_n = 1000
d = 100
trials = 10
steps = 1000000

def grads(X, y, a):
	result = np.dot(X, a)
	return 2 * np.dot(X.T, y - result)

section_a = []
section_b = [[] for i in range(7)]
section_b_train = [[] for i in range(7)]
section_c = [[] for i in range(3)]
section_c_train = [[] for i in range(3)]
section_d = [[[[] for _ in range(steps//train_n)] for _ in range(2)] for _ in range(3)]
section_e = [[[] for i in range(7)] for _ in range(2)]

lambs_b = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
lrs_c = [0.00005, 0.0005, 0.005]
lrs_d = [0.00005, 0.005]
rs = [0, 0.1, 0.5, 1, 10, 20, 30]

for _ in range(trials):
	X_train = np.random.normal(0,1, size=(train_n,d))
	a_true = np.random.normal(0,1, size=(d,1))
	y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
	X_test = np.random.normal(0,1, size=(test_n,d))
	y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))

	#part a
	a = np.dot(np.linalg.inv(X_train), y_train)
	section_a.append(np.linalg.norm(X_test.dot(a) - y_test) / np.linalg.norm(y_test))

	#part b
	for i in range(7):
		a = np.linalg.inv(X_train.T.dot(X_train) + lambs_b[i] * np.eye(d)).dot(X_train.T).dot(y_train)
		section_b[i].append(np.linalg.norm(X_test.dot(a) - y_test) / np.linalg.norm(y_test))
		section_b_train[i].append(np.linalg.norm(X_train.dot(a) - y_train) / np.linalg.norm(y_train))

	#part c
	for i in range(3):
		a = np.zeros((d, 1))
		order = np.random.permutation(train_n)
		for e in range(steps//train_n):
			for s in range(train_n):
				a += lrs_c[i] * grads(X_train[order[s]].reshape(1, d), y_train[order[s]], a)
		section_c[i].append(np.linalg.norm(X_test.dot(a) - y_test) / np.linalg.norm(y_test))
		section_c_train[i].append(np.linalg.norm(X_train.dot(a) - y_train) / np.linalg.norm(y_train))

	#part d
	for i in range(2):
		a = np.zeros((d, 1))
		order = np.random.permutation(train_n)
		for e in range(steps//train_n):
			for s in range(train_n):
				a += lrs_d[i] * grads(X_train[order[s]].reshape(1, d), y_train[order[s]], a)
			section_d[0][i][e].append(np.linalg.norm(X_train.dot(a) - y_train) / np.linalg.norm(y_train))
			section_d[1][i][e].append(np.linalg.norm(X_test.dot(a) - y_test) / np.linalg.norm(y_test))
			section_d[2][i][e].append(np.linalg.norm(a))

	for i in range(7):
		a = np.random.normal(0, 1, size=(d, 1))
		a = a * rs[i] / np.linalg.norm(a)
		order = np.random.permutation(train_n)
		for e in range(steps//train_n):
			for s in range(train_n):
				a += lrs_c[0] * grads(X_train[order[s]].reshape(1, d), y_train[order[s]], a)
		section_e[0][i].append(np.linalg.norm(X_train.dot(a) - y_train) / np.linalg.norm(y_train))
		section_e[1][i].append(np.linalg.norm(X_test.dot(a) - y_test) / np.linalg.norm(y_test))


print("(a)", np.mean(section_a))
print("(b)", np.mean(section_b, axis=-1))
print("(c)", np.mean(section_c, axis=-1))
print("(c) train", np.mean(section_c_train, axis=-1))
print("(d)", np.mean(section_d, axis=-1))
print("(e)", np.mean(section_e, axis=-1))

test, = plt.plot(lambs_b, np.mean(section_b, axis=-1), color='red', lw=2)
train, = plt.plot(lambs_b, np.mean(section_b_train, axis=-1), color='blue', lw=2)
plt.legend([train, test], ['Train Loss', 'Test Loss'])
plt.ylabel('Test Loss')
plt.xlabel('Regularization Parameter')
plt.xscale('log')
plt.savefig('2b.png', bbox_inches='tight')
plt.clf()

plt.plot(range(1, steps + 1, 100), np.mean(section_d, axis=-1)[0][0], color='blue', lw=2)
plt.plot(range(1, steps + 1, 100), [np.linalg.norm(X_test.dot(a_true) - y_test) / np.linalg.norm(y_test) for _ in range(len(range(1, steps + 1, 100)))], color='green', lw=2)
plt.ylabel('Training Loss')
plt.xlabel('Iteration Number')
plt.savefig('2di1.png', bbox_inches='tight')
plt.clf()

plt.plot(range(1, steps + 1, 100), np.mean(section_d, axis=-1)[0][1], color='blue', lw=2)
plt.plot(range(1, steps + 1, 100), [np.linalg.norm(X_test.dot(a_true) - y_test) / np.linalg.norm(y_test) for _ in range(len(range(1, steps + 1, 100)))], color='green', lw=2)
plt.ylabel('Training Loss')
plt.xlabel('Iteration Number')
plt.savefig('2di2.png', bbox_inches='tight')
plt.clf()

plt.plot(range(1, steps + 1, 100), np.mean(section_d, axis=-1)[1][0], color='blue', lw=2)
plt.ylabel('Test Loss')
plt.xlabel('Iteration Number')
plt.savefig('2dii1.png', bbox_inches='tight')
plt.clf()

plt.plot(range(1, steps + 1, 100), np.mean(section_d, axis=-1)[1][1], color='blue', lw=2)
plt.ylabel('Test Loss')
plt.xlabel('Iteration Number')
plt.savefig('2dii2.png', bbox_inches='tight')
plt.clf()

plt.plot(range(1, steps + 1, 100), np.mean(section_d, axis=-1)[2][0], color='blue', lw=2)
plt.ylabel('L2 Norm of a')
plt.xlabel('Iteration Number')
plt.savefig('2diii1.png', bbox_inches='tight')
plt.clf()

plt.plot(range(1, steps + 1, 100), np.mean(section_d, axis=-1)[2][1], color='blue', lw=2)
plt.ylabel('L2 Norm of a')
plt.xlabel('Iteration Number')
plt.savefig('2diii2.png', bbox_inches='tight')
plt.clf()

train, = plt.plot(rs, np.mean(section_e, axis=-1)[0], color='blue', lw=2)
test, = plt.plot(rs, np.mean(section_e, axis=-1)[1], color='red', lw=2)
plt.legend([train, test], ['Train Loss', 'Test Loss'])
plt.ylabel('Loss')
plt.xlabel('Size of Sphere')
plt.savefig('2e.png', bbox_inches='tight')
plt.clf()