import pickle
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Tmall/Nowplaying')
parser.add_argument('--sample_num', type=int, default=12)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num

seq = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))

if dataset == 'diginetica':
    num = 43098
elif dataset == "Tmall":
    num = 40728
elif dataset == "Nowplaying":
    num = 60417
else:
    num = 3
    ''''''
long = {}
for l in seq:
    i = len(l)
    if i in long:
        long[i] = long[i] + 1
    else:
        long[i] = 1
''''''
relation = []
neighbor = [] * num

all_test = set()


'''
for i in range(len(seq)):
    data = seq[i]
    for k in range(1, 4):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1
'''
for i in range(len(seq)):
    if len(seq[i]) < 3:
        continue
    data = seq[i]
    if len(seq[i]) == 3:
        relation.append([(data[0], data[1]), (data[1], data[2])])
    if len(seq[i]) > 3:
        for k in range(len(data)-3):
            relation.append([(data[k], data[k+1]), (data[k+1], data[k+2])])
            relation.append([(data[k], data[k + 1]), (data[k + 2], data[k + 3])])
            relation.append([(data[k + 1], data[k + 2]), (data[k], data[k + 1])])
            relation.append([(data[k + 2], data[k + 3]), (data[k], data[k + 1])])

max = 0
my_adj = dict()
for group in relation:
    if group[0] not in my_adj.keys():
        my_adj[group[0]] = defaultdict(set)
        my_adj[group[0]][group[1]] = 1
    else:
        if my_adj[group[0]][group[1]] == ():
            my_adj[group[0]][group[1]] = 1
        else:
            my_adj[group[0]][group[1]] =+ 1



weight = [[] for _ in range(num)]
'''
for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]
'''
adj = dict()
for k, v in my_adj.items():
    tup = list(v.keys())
    num = list(v.values())
    comb = zip(tup, num)
    zipped = list(comb)
    x = [i for i in sorted(zipped, reverse=True, key=lambda x:[1])]
    x = x[:3]
    d = dict(x)
    adj[k] = dict()
    adj[k] = d


for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]


pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('datasets/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))
