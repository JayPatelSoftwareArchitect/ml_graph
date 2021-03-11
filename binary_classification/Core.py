
import sys
sys.path.append('/home/jay/Main/tmp_prj/ml_graph/')
print(sys.path)
from TModel import Model
import tensorflow as tf
from Training import Training

import csv
x_data = []
y_data = []
r_index = []
c=0
with open('/home/jay/Main/tmp_prj/ml_graph/binary_classification/sonar.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        f = ""
        v = ""
        tmp = []
        for i in range(0, len(row[0])):
            if row[0][i] == ",":
                tmp.append(float(f))
                f = ""
            elif row[0][i] == "R" or row[0][i] == "M":
                v = row[0][i]
            else:        
                f += row[0][i]
        if v == "R":
            y_data.append(0)
        else: 
            y_data.append(1)
        x_data.append(tmp)
        r_index.append(c)
        c+=1
        # print(', '.join(row))

import random
random.shuffle(r_index)
x = []
y = []
for j in range(0, len(r_index)):
    index__ = r_index[j]
    x.append(x_data[index__])
    y.append(y_data[index__])
model = Model(NumLayer=5, NumTensor=[10, 6, 4, 4, 2])
model.setData(x, y, x, y)
training = Training(model)
training.fit()