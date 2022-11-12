import pandas as pd
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data",help="dateset")
args = parser.parse_args()
data = args.data

df = pd.read_csv(data,header=None)
df.columns = ['y'] + ['x'+str(i+1) for i in range(len(df.columns)-1)]
#compute gaussian parameters for the dataset
def get_gaussian_parameters():
    parameters = {}
    class_labels = df['y'].unique()
    columns = list(df.columns)[1:]
    for c in class_labels:
        class_parameters = {}
        for col in columns:
            class_parameters[col+'_mean'] = df[col][df['y']==c].mean()
            class_parameters[col+'_variance'] = df[col][df['y']==c].var()
        class_parameters['p'] = len(df[df['y']==c])/len(df)
        parameters[c] = class_parameters
    return parameters

#function to compute gaussian estimate given a class for 1 instance
def gaussian_estimate(c,attributes):
    gaussian_parameters = get_gaussian_parameters()[c]
    probability = 1
    for atr in attributes.keys():
        x = attributes[atr]
        mu = gaussian_parameters[atr+'_mean']
        var = gaussian_parameters[atr+'_variance']
        estimate = (1/math.sqrt(2*(math.pi)*var))*(math.exp((-(x-mu)**2)/(2*var)))
        probability *= estimate
    probability *= gaussian_parameters['p']
    return probability

error = 0
#loop through each instance
for instance in range(len(df)):
    true_class = df.iloc[instance,0]
    class_probabilities = {}
# for each class
    for c in df['y'].unique():
        instance_values = df.iloc[instance,1:].to_dict()
        #store the likelihood of the class in the dictionary
        class_probabilities[c] = gaussian_estimate(c,instance_values)

# get the class label for which the likelihood is maximum
    pred_class = max(class_probabilities,key=class_probabilities.get)

    if pred_class!=true_class:
        error+=1

for key,val in get_gaussian_parameters().items():
    print(*val.values(),sep=",")
print(error)
