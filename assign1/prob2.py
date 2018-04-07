import numpy as np
import hashlib

ntrials = 10
ntables = 4
ncounters = 256
conservupdates = True
if conservupdates:
	print("Conservative Updates: On")
else:
	print("Conservative Updates: Off")

def gethashes(number, nhashes=1, trial=0):
	glob = str(number) + str(trial)
	result = hashlib.md5(glob.encode('utf-8')).hexdigest()
	for i in range(nhashes):
		yield int(result[i * 2:(i + 1) * 2], 16)

stream = np.zeros(87925, dtype=np.int32)
index = 0
for j in range(9):
	for k in range(1, 1001):
		for _ in range(j + 1):
			stream[index] = j * 1000 + k
			index += 1
for j in range(50):
	for _ in range((j + 1) ** 2):
		stream[index] = 9001 + j
		index += 1

def runanorientation(stream, ntrials, ntables, ncounters, conservupdates):
	heavyhittercounts = []
	topcounts = []

	for i in range(ntrials):
		table = np.zeros((ntables, ncounters), dtype=np.int32)
		for j in range(stream.shape[0]):
			result = np.array(list(gethashes(stream[j], ntables, i)), dtype=np.int32)
			if conservupdates:
				firstmask = np.zeros(table.shape)
				firstmask[np.arange(ntables), result] = 1
				mask = np.logical_and(firstmask, table==np.min(table[np.arange(ntables), result]))
				table[mask] += 1
			else:
				table[np.arange(ntables), result] += 1
		heavyhitters = 0
		for j in range(1, 9051):
			result = np.array(list(gethashes(j, ntables, i)), dtype=np.int32)
			if np.min(table[np.arange(ntables), result]) > stream.shape[0] / 100.:
				heavyhitters += 1
		result = np.array(list(gethashes(9050, ntables, i)), dtype=np.int32)
		topcounts.append(np.min(table[np.arange(ntables), result]))
		heavyhittercounts.append(heavyhitters)

	print("Heavy Hitter (HH) Counts Per Trial:", heavyhittercounts)
	print("9050 Count Per Trial:", topcounts)
	print()
	print("Average HH Count:", np.mean(heavyhittercounts), "Average 9050 Count:", np.mean(topcounts))
	print()
	print()

print(25 * "=" + "Increasing Order" + 25 * "=")
runanorientation(stream, ntrials, ntables, ncounters, conservupdates)

print(25 * "=" + "Decreasing Order" + 25 * "=")
stream = np.flip(stream, 0)
runanorientation(stream, ntrials, ntables, ncounters, conservupdates)

print(27 * "=" + "Random Order" + 27 * "=")
np.random.shuffle(stream)
runanorientation(stream, ntrials, ntables, ncounters, conservupdates)