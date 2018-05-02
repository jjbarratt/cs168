import numpy as np
import matplotlib.pyplot as plt

def lsrecover(X, Y):
	xminus = X - np.mean(X)
	return xminus.dot(Y - np.mean(Y)) / xminus.dot(xminus)

def pcarecover(X, Y):
	result = np.linalg.eig([[X.dot(X), X.dot(Y)], [X.dot(Y), Y.dot(Y)]])
	principalvec = result[1][np.argmax(result[0])]
	return -principalvec[1]/principalvec[0]

#part b
pca = []
ls = []
for i in range(100):
	X = np.random.uniform(size=(1000))
	Y = np.random.uniform(size=(1000))
	pca.append(pcarecover(X, Y))
	ls.append(lsrecover(X, Y))

print(np.mean(pca))
print(np.mean(ls))

#part c
X = np.array([(i + 1)/1000 for i in range(1000)])
c = [i/20 for i in range(11)]
data = np.zeros((2, 2, 330))
trials = 30
for i in range(len(c)):
	for t in range(trials):
		Y = np.random.normal(size=(1000,)) * c[i] + 2 * X
		data[0][0][i * trials + t] = c[i]
		data[1][0][i * trials + t] = c[i]
		data[0][1][i * trials + t] = lsrecover(X, Y)
		data[1][1][i * trials + t] = pcarecover(X, Y)

plt.scatter(data[0][0], data[0][1], label="Least Squares", color="blue")
plt.scatter(data[1][0], data[1][1], label="PCA", color="red")
plt.xlabel("Multiplier of Noise")
plt.ylabel("Estimated Slope")
plt.legend()
plt.savefig("2c.png")
plt.clf()

#part d
c = [i/20 for i in range(11)]
data = np.zeros((2, 2, 330))
trials = 30
for i in range(len(c)):
	for t in range(trials):
		base = np.array([(i + 1)/1000 for i in range(1000)])
		X = np.random.normal(size=(1000,)) * c[i] + base
		Y = np.random.normal(size=(1000,)) * c[i] + 2 * base
		data[0][0][i * trials + t] = c[i]
		data[1][0][i * trials + t] = c[i]
		data[0][1][i * trials + t] = lsrecover(X, Y)
		data[1][1][i * trials + t] = pcarecover(X, Y)

plt.scatter(data[0][0], data[0][1], label="Least Squares", color="blue")
plt.scatter(data[1][0], data[1][1], label="PCA", color="red")
plt.xlabel("Multiplier of Noise")
plt.ylabel("Estimated Slope")
plt.legend()
plt.savefig("2d.png")