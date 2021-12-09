import pandas as pd
import numpy as np
import math
import argparse
import sys

# File for verification purpose
sys.stdout = open('my_sol.csv','w')


parser = argparse.ArgumentParser()
parser.add_argument("--data",help="dateset")
parser.add_argument("--eta",help="eta",type=float)
parser.add_argument("--iterations",help="iterations",type=int)
args = parser.parse_args()
data =  args.data
eta = args.eta
iterations = args.iterations
df = pd.read_csv(data,names = ['a','b','y'])


# initializations

w_bias_h1 = 0.2
w_a_h1 = -0.3
w_b_h1 = 0.4
w_bias_h2 = -0.5
w_a_h2 = -0.1
w_b_h2 = -0.4
w_bias_h3 = 0.3
w_a_h3 = 0.2
w_b_h3 = 0.1
w_bias_o = -0.1
w_h1_o = 0.1
w_h2_o = 0.3
w_h3_o = -0.4



## sigmoid
def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return (1 / (1 + math.exp(-x)))

print('a,b,h1,h2,h3,o,t,delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o')
print('-,-,-,-,-,-,-,-,-,-,-,{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o))

for _ in range(iterations):
    for i in range(len(df)):
        # forward pass

        net_h1 = df['a'][i]*w_a_h1 + df['b'][i]*w_b_h1 + 1*w_bias_h1
        o_h1 =  (sigmoid(net_h1))

        net_h2 = df['a'][i]*w_a_h2 + df['b'][i]*w_b_h2 + 1*w_bias_h2
        o_h2 =  (sigmoid(net_h2))

        net_h3 = df['a'][i]*w_a_h3 + df['b'][i]*w_b_h3 + 1*w_bias_h3
        o_h3 =  (sigmoid(net_h3))

        net_o = o_h1*w_h1_o + o_h2*w_h2_o + o_h3*w_h3_o + 1*w_bias_o
        o = (sigmoid(net_o))
        t = df['y'][i]

        # backward pass

        # computing deltas
        delta_o = o*(1-o)*(t-o)
        delta_h1 = o_h1*(1-o_h1)*(w_h1_o*delta_o)
        delta_h2 = o_h2*(1-o_h2)*(w_h2_o*delta_o)
        delta_h3 = o_h3*(1-o_h3)*(w_h3_o*delta_o)

        # updating weights

        w_bias_h1 = (w_bias_h1) + (eta*1*delta_h1)
        w_a_h1 = (w_a_h1) + (eta*df['a'][i]*delta_h1)
        w_b_h1 = (w_b_h1) + (eta*df['b'][i]*delta_h1)

        w_bias_h2 = (w_bias_h2) + (eta*1*delta_h2)
        w_a_h2 = (w_a_h2) + (eta*df['a'][i]*delta_h2)
        w_b_h2 = (w_b_h2) + (eta*df['b'][i]*delta_h2)

        w_bias_h3 = (w_bias_h3) + (eta*1*delta_h3)
        w_a_h3 = (w_a_h3) + (eta*df['a'][i]*delta_h3)
        w_b_h3 = (w_b_h3) + (eta*df['b'][i]*delta_h3)

        w_bias_o = (w_bias_o) + (eta*1*delta_o)

        w_h1_o = (w_h1_o) + (eta*o_h1*delta_o)
        w_h2_o = (w_h2_o) + (eta*o_h2*delta_o)
        w_h3_o = (w_h3_o) + (eta*o_h3*delta_o)

        print('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(df['a'][i],df['b'][i],o_h1,o_h2,o_h3,o,t,delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o))
