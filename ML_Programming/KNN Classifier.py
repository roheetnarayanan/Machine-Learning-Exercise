import pandas as pd
import math
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data",help="dateset")
parser.add_argument("--k",help="Neighbours",type=int)
args = parser.parse_args()
data = args.data
k = args.k

df = pd.read_csv(data,names=['y','x1','x2'])

case_base = {0:list(df.iloc[0,1:])}

def l2_distance(q,x):
    return math.sqrt( sum([(q[i]-x[i])**2 for i in range(len(q))]))

#build case base
for i in range(1,len(df)):
    dist = {}
    for case in list(case_base.keys()):
        dist[case] = l2_distance(case_base[case],list(df.iloc[i,1:]))
    NN = min(dist,key=dist.get)
    if df.iloc[NN,0]!=df.iloc[i,0]:
        case_base[i] = list(df.iloc[i,1:])


# KNN based on case base
def KNN(i,k):
    #compute the distance from i to case in case_base
    dist = {}
    for instance in case_base.keys():
        dist[instance] = l2_distance(i,case_base[instance])

    distance = dict(sorted([(k,v) for k, v in dist.items()], key=lambda x: x[1])[:k])

    #compute the weight of i to case in case_base
    weights = {}
    class_a,class_b = 0,0
    for instance in distance.keys():
        dk_nn = max(distance,key=distance.get)
        d1_nn = min(distance,key=distance.get)
        if dist[dk_nn]==dist[d1_nn]:
            weights[instance] = 1
        else:
            weights[instance] = (dist[dk_nn]-dist[instance])/(dist[dk_nn]-dist[d1_nn])

        if df.iloc[instance,0]=='A':
            class_a += weights[instance]
        else:
            class_b += weights[instance]

    if class_a>class_b:
        return 'A'
    else:
        return 'B'


# compute KNN of each instance to the case base
error = 0
for i in range(len(df)):
    if i in case_base:
        pass
    else:
        knn = KNN(list(df.iloc[i,1:]),k)
        if knn != df.iloc[i,0]:
            error+=1

print(error)
for instance in case_base.keys():
    print(*list(df.iloc[instance]),sep=",")
