from TModel import Model
import tensorflow as tf
from Training import Training
from ActivationFn import Activation
import hashlib
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample_x = []
sample_y = []
for i in range(0, 10):
    temp_ = ""
    for j in range(0, len(x_train[i])):
        for k in range(0, len(x_train[i][j])):
            temp_ += str(x_train[i][j][k])
            temp_ += ":"
    sample_x.append(temp_)
    sample_y.append(y_train[i])
op = []
for i in range(0, len(sample_x)):
    hash_obj = hashlib.md5(sample_x[i].encode())
    op.append(hash_obj.hexdigest())

model = Model(NumLayer=10, NumTensor=[28, 28, 14, 14, 10, 10, 10, 10, 10, 10])
training = Training(model)
output_sample = []
for i in range(0,10) :
    output_sample.append(i)

training.sample(op[0], output_sample)
training.setActivationFunction(Activation.relu)
training.fit(op,sample_y)
model.print()
#For each tensor take all previous connected weights and add them up add bais apply activation and pass that value to next connected weights.
