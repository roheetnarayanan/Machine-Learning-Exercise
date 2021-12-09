import sys
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data",help="dateset")
parser.add_argument("--eta", help="value for eta", type = float)
parser.add_argument("--threshold", help="value for threshold", type = float)
args = parser.parse_args()

data =  args.data
eta = args.eta
threshold = args.threshold

df = pd.read_csv(data, names = ['x1','x2','y'])



thresholdreached = True
w0, w1, w2 = 0, 0, 0
iterations = 0
prev_sse = 0
while thresholdreached == True:
    sse = 0

    grad0, grad1, grad2 = 0, 0, 0
    for i in range(len(df)):
        yhat =  w0*1 + w1*df['x1'][i] + w2*df['x2'][i]
        error = float(df['y'][i] - yhat)
        sse += float(error)**2
        grad0 += error*1
        grad1 += error*df['x1'][i]
        grad2 += error*df['x2'][i]

    print('{},{},{},{},{}'.format(iterations, w0,w1,w2,sse))

    if iterations==0:
        pass
    elif (prev_sse-sse)<threshold:
        thresholdreached = False
        break
    w0 = w0 + float(eta*grad0)
    w1 = w1 + float(eta*grad1)
    w2 = w2 + float(eta*grad2)
    prev_sse = sse
    iterations +=1
