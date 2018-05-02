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
data = np.array([list(map(maptonum, elem[3:])) for elem in content], dtype=np.float32)
mask = mode(data, axis=0)[0]
X = (data != mask).astype(np.float32)
X -= X.mean(axis=1, keepdims=True)

pcathing = PCA(n_components=3)
pcathing.fit(X)
components = pcathing.components_
projection = X.dot(components.T) #3d projection of X onto PCA vectors
places = np.unique(metadata[:,2])
sexes = ['1', '2']

#part b
places = np.unique(metadata[:,2])
for place in places:
	index = metadata[:,2] == place
	plt.scatter(projection[index,0], projection[index,1], label=place, s=10)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Population')
plt.savefig('1b.png', bbox_inches='tight')
plt.clf()

#part d
for sex in sexes:
	index = metadata[:,1] == sex
	plt.scatter(projection[index,0], projection[index,2], label='Male' if sex == '1' else 'Female', s=10)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 3')
plt.legend(title='Sex')
plt.savefig('1d.png', bbox_inches='tight')
plt.clf()

#part f
plt.scatter(np.arange(components.shape[1]), np.abs(components[2]))
plt.xlabel('Nucleobase Index')
plt.ylabel('Absolute Value of Principal Component 3')
plt.savefig('1f.png', bbox_inches='tight')
plt.clf()