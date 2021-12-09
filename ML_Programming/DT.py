import pandas as pd
import numpy as np
import math
import argparse
import sys

# File for verification purpose
sys.stdout = open('my_sol.txt','w')
parser = argparse.ArgumentParser()
parser.add_argument("--data",help="dateset")
args = parser.parse_args()
data =  args.data


df = pd.read_csv(data)
names=['att'+str(i) for i in range(len(df.columns)-1)]
names.append('y')
df = pd.read_csv(data,names=names)
log_base = len(df.iloc[:,-1].unique())
eps = np.finfo(float).eps



# function to get the entropy of a node S
def get_entropy(df):

    Class = df.keys()[-1]
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*math.log(fraction,log_base)
    return entropy

def get_attr_entropy(df,attribute):
    #Function to calculate the entropy over all features in the table.
    Class = df.keys()[-1]
    target_variables = df[Class].unique()
    variables = df[attribute].unique()
    weighted_entropy = 0
    # looping for each attr value
    for variable in variables:
        entropy = 0
        # looping for each class
        for target_variable in target_variables:
                numerator = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
                denominator = len(df[attribute][df[attribute]==variable])
                fraction = numerator/(denominator)
                entropy += -fraction*math.log(fraction+eps,log_base)
        # compute the weighted _entropy
        weights = denominator/len(df)
        weighted_entropy += -weights*entropy
    return abs(weighted_entropy)

def get_best_feature(df):
    Entropy_att = []
    IG = {}
    for key in df.keys()[:-1]:

        IG[key] = get_entropy(df)-get_attr_entropy(df,key)
    return max(IG, key= IG.get)


def get_subtable(df, node, value):
    # get the subtable when equals the given attribute and attr value
    # deleting the col as it should not be used again
    return df[df[node] == value].drop(node,axis=1).reset_index(drop=True)

def get_most_common(df):
    most_list = list(df.iloc[:,-1])
    return max(set(most_list),key = most_list.count)



def growtree(df,depth=None):
    #Here we build the decision tree
    if depth==None:
        depth = 0
        print('{},root,{},no_leaf'.format(depth,get_entropy(df)))
        depth = depth+1

    #Get feature with maximum information gain
    node = get_best_feature(df)

    attValue = np.unique(df[node])

    for value in attValue:

        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['y'],return_counts=True)

        if len(counts)==1:#Check if subset is pure
            print('{},{}={},{},{}'.format(depth,node,value,get_entropy(subtable),clValue[0]))

        elif len(df.columns)==1:# check if no more attr are left to split
            print('{},{}={},{},{}'.format(depth,node,value,get_entropy(subtable),get_most_common(subtable)))

        else:
            print('{},{}={},{},no_leaf'.format(depth,node,value,get_entropy(subtable)))
            growtree(df=subtable,depth=depth+1) #Calling the function recursively

    return None
growtree(df=df)
