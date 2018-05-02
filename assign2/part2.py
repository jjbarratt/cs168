import matplotlib.pyplot as plt
import numpy as np
import csv
import warnings
import math

def makeHeatMap(data, names, color, outputFileName):
    #to catch "falling back to Agg" warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #code source: http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
        fig, ax = plt.subplots()
        #create the map w/ color bar legend
        heatmap = ax.pcolor(data, cmap=color)
        cbar = plt.colorbar(heatmap)

        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(range(1, 21))
        ax.set_yticklabels(names)

        plt.tight_layout()

        plt.savefig(outputFileName, format = 'png')
        plt.close()

def cosine(first, second):
    first, second = dict(first), dict(second)

    numer_sum = 0
    second_squared = 0
    first_squared = 0
    for key in second:
        if key in first:
            numer_sum += (second[key] * first[key])
            first_squared += (first[key] ** 2)
            del first[key]
        second_squared += (second[key] ** 2)

    for key in first:
        first_squared += (first[key] ** 2)

    return numer_sum / np.sqrt(second_squared * first_squared)

data = []
with open('data50.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(list(map(int, row)))
data = np.array(data)
dict1 = {}

maxword = 0
groups = []
with open('label.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        groups.append(int(row[0]))
for i in range(data.shape[0]):
    group = groups[data[i][0] - 1]
    article = data[i][0]
    if group in dict1:
        temp_dict = dict1[group]
        word = data[i][1]
        if word > maxword:
            maxword = word
        if article not in temp_dict:
            temp_dict[article] = {data[i][1]: data[i][2]}
            dict1[group] = temp_dict
        elif word in temp_dict[article]:
            temp_dict[article][word] += data[i][2]
            dict1[group] = temp_dict
        else:
            temp_dict[article][word] = data[i][2]
            dict1[group] = temp_dict
    else:
        dict1[group] = {article: {data[i][1]: data[i][2]}}

names = []
with open('groups.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        names.append(row[0])

def nearestneighbor(article, dict1):
    maxsim = 0
    bestindex = 0
    for group in dict1:
        for art in dict1[group]:
            if article == dict1[group][art]:
                continue
            similarity = cosine(dict1[group][art], article)
            if similarity > maxsim:
                maxsim = similarity
                bestindex = group
    return bestindex

# code for finding naive nearest neighbors without dimension reduction:
final_data = np.zeros((20, 20))
for i in range(final_data.shape[0]):
    for article in dict1[i + 1]:
        final_data[i,nearestneighbor(dict1[i + 1][article], dict1) - 1] += 1
makeHeatMap(final_data, names, 'Blues', 'nnsim.png')
print("Final Accuracy:", np.sum(np.multiply(np.eye(20), final_data))/np.sum(final_data))

def cosinedense(x, y):
    return np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))

def nndense(article, dict1, groupnum, artnum):
    maxsim = 0
    bestindex = 1
    for group in dict1:
        for art in dict1[group]:
            if group == groupnum and art == artnum:
                continue
            similarity = cosinedense(dict1[group][art], article)
            if similarity > maxsim:
                maxsim = similarity
                bestindex = group
    return bestindex

#dimension reduction code:
d = 100
M = np.random.randn(d, maxword + 1)

for group in dict1:
    for art in dict1[group]:
        dict1[group][art] = sum([M[:,i] * dict1[group][art][i] for i in dict1[group][art]])

final_data = np.zeros((20, 20))
for i in range(final_data.shape[0]):
    for article in dict1[i + 1]:
        final_data[i,nndense(dict1[i + 1][article], dict1, i + 1, article) - 1] += 1
makeHeatMap(final_data, names, 'Blues', str(d) + 'nnsim.png')
print("Final Accuracy:", np.sum(np.multiply(np.eye(20), final_data))/np.sum(final_data))