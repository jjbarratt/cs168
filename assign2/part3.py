import matplotlib.pyplot as plt
import numpy as np
import csv
import warnings
import math
import copy

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

def anyequal(article, elem, l):
    for i in range(l):
        if np.all(article[i] == elem[i]):
            return True
    return False

def nndense(article, dict1, groupnum, artnum, origdict, l):
    maxsim = 0
    bestindex = 1
    nsd = 0
    for group in dict1:
        for art in dict1[group]:
            if group == groupnum and art == artnum:
                continue
            if not anyequal(article, dict1[group][art], l):
                continue
            nsd += 1
            similarity = cosine(origdict[group][art], origdict[groupnum][artnum])
            if similarity > maxsim:
                maxsim = similarity
                bestindex = group
    return bestindex, nsd

origdict = dict1
l = 128
M = [None for i in range(l)]

#dimension reduction code:
for d in range(5, 21):
    for i in range(l):
        M[i] = np.random.randn(d, maxword + 1)
    dict1 = copy.deepcopy(origdict)
    for group in dict1:
        for art in dict1[group]:
            dict1[group][art] = [np.sign(sum([M[j][:,i] * dict1[group][art][i] for i in dict1[group][art]])) for j in range(l)]

    final_data = np.zeros((20, 20))
    totalnsd = 0
    for i in range(final_data.shape[0]):
        for article in dict1[i + 1]:
            group, nsd = nndense(dict1[i + 1][article], dict1, i + 1, article, origdict, l)
            totalnsd += nsd
            final_data[i,group - 1] += 1
    print("Final Accuracy:", np.sum(np.multiply(np.eye(20), final_data))/np.sum(final_data), " Average Sd size:", totalnsd/1000)