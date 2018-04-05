import numpy as np 
import matplotlib.pyplot as plt

N = 200000
n_trials = 30

def pickN(N, n_picks = 1):
	hist = np.zeros(N, dtype=np.int32)
	for i in range(N):
		picks = np.random.choice(N, n_picks)
		hist[picks[np.argmin(hist[picks])]] += 1
	return np.max(hist)

def strat1(N):
	# return pickN(N, 1)
	return np.max(np.unique(np.random.choice(N, N), return_counts=True)[1])

def strat2(N):
	return pickN(N, 2)

def strat3(N):
	return pickN(N, 3)

def strat4(N, n_buckets=2):
	hist = np.zeros(N, dtype=np.int32)
	for i in range(N):
		picks = np.array([np.random.choice(N//n_buckets) + i * N//n_buckets for i in range(n_buckets)])
		hist[picks[np.argmin(hist[picks])]] += 1
	return np.max(hist)

bins = np.zeros((N, 4))
funcs = [strat1, strat2, strat3, strat4]
for i in range(len(funcs)):
	for j in range(n_trials):
		bins[funcs[i](N) - 1, i] += 1
	print("Done with strat" + str(i + 1))

def plot_histogram(bins, filename = None):
	assert bins.shape == (200000,4), "Input bins must be a numpy array of shape (max_bin_population, num_strategies)"
	assert np.array_equal(np.sum(bins, axis = 0),(np.array([30,30,30,30]))), "There must be 30 runs for each strategy"

	thresh =  max(np.nonzero(bins)[0])+3
	n_bins = thresh
	bins = bins[:thresh,:]
	print("\nPLOTTING: Removed empty tail. Only the first non-zero bins will be plotted\n")

	ind = np.arange(n_bins) 
	width = 1.0/6.0

	fig, ax = plt.subplots()
	rects_strat_1 = ax.bar(ind + width, bins[:,0], width, color='yellow')
	rects_strat_2 = ax.bar(ind + width*2, bins[:,1], width, color='orange')
	rects_strat_3 = ax.bar(ind + width*3, bins[:,2], width, color='red')
	rects_strat_4 = ax.bar(ind + width*4, bins[:,3], width, color='k')

	ax.set_ylabel('Number Occurrences in 30 Runs')
	ax.set_xlabel('Number of Balls In The Most Populated Bin')
	ax.set_title('Histogram: Load on Most Populated Bin For Each Strategy')

	ax.set_xticks(ind)
	ax.set_xticks(ind+width*3, minor = True)
	ax.set_xticklabels([str(i+1) for i in range(0,n_bins)], minor = True)
	ax.tick_params(axis=u'x', which=u'minor',length=0)

	ax.legend((rects_strat_1[0], rects_strat_2[0], rects_strat_3[0], rects_strat_4[0]), ('Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4'))
	plt.setp(ax.get_xmajorticklabels(), visible=False)
	
	if filename is not None: plt.savefig(filename+'.png', bbox_inches='tight')

	plt.show()

plot_histogram(bins, "fig1")