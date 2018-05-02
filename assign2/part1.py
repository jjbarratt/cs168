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

def jaccard(first, second):
    first, second = dict(first), dict(second)

    denominator = 0
    numerator = 0
    for key in second:
        if key in first:
            numerator += min(second[key], first[key])
            denominator += max(second[key], first[key])
            del first[key]
        else:
            denominator += second[key]

    for key in first:
        denominator += first[key]
    return numerator / denominator

def L2(first, second):
    first, second = dict(first), dict(second)

    total_sum = 0;
    for key in second:
        if key in first:
            difference = second[key] - first[key]
            squared = difference ** 2
            total_sum += squared
            del first[key]
        else:
            total_sum += (second[key] ** 2)

    for key in first:
        total_sum += (first[key] ** 2)

    return (-1 * math.sqrt(total_sum))

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

fns = [jaccard, L2, cosine]
fnnames = ['jaccard', 'l2', 'cosine']
index = 0
for fn in fns:
    final_data = np.eye(20)
    for i in range(final_data.shape[0]):
        for j in range(final_data.shape[1]):
            final_data[i][j] = np.mean([fn(first, second) for _,first in dict1[i + 1].items()
            	for _,second in dict1[j + 1].items()])
    makeHeatMap(final_data, names, 'Blues', fnnames[index] + '.png')
    index += 1