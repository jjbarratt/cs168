from sklearn.decomposition import PCA
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt

content = None
with open('data.txt') as f:
    content = f.readlines()
content = [x.split() for x in content]

dictthing = {'A':1, 'C': 2, 'G':3, 'T':4}
def maptonum(nucleotide):
	return dictthing[nucleotide]

metadata = np.array([elem[:3] for elem in content])
data = np.array([list(map(maptonum, elem[3:])) for elem in content], dtype=np.int32)

X = np.zeros((data.shape[0], data.shape[1] * 4))
for i in range(data.shape[0]):
	X[i][np.arange(data.shape[1]) * 4 + data[i] - 1] = 1
X -= X.mean(axis=1, keepdims=True)

pcathing = PCA(n_components=4)
pcathing.fit(X)
components = pcathing.components_
projection = X.dot(components.T) #4d projection of X onto PCA vectors
places = np.unique(metadata[:,2])
sexes = ['1', '2']

#part h
places = np.unique(metadata[:,2])
for place in places:
	index = metadata[:,2] == place
	plt.scatter(projection[index,0], projection[index,1], label=place, s=10)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Population')
plt.savefig('1h.png', bbox_inches='tight')
plt.clf()

#part i
places = np.unique(metadata[:,2])
for place in places:
	index = metadata[:,2] == place
	plt.scatter(projection[index,0], projection[index,3], label=place, s=10)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 4')
plt.legend(title='Population')
plt.savefig('1i.png', bbox_inches='tight')
plt.clf()