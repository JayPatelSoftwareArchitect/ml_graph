from TModel import Model
import tensorflow as tf
from Training import Training
from LogisticRegression import LogisticRegression
import hashlib
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample_x = []
sample_y = []


model = Model(NumLayer=10, NumTensor=[32, 28, 14, 14, 10, 10, 10, 10, 10, 10])
model.setData(x_train[1:100], y_train[1:100], x_test[1:100], y_test[1:100], op='md5')
training = Training(model)

output_sample = []
for i in range(0,10) :
    output_sample.append(i)

training.sample(x_train[0], output_sample)
training.setActivationFunction(LogisticRegression)
training.fit()

model.print()
#For each tensor take all previous connected weights and add them up add bais apply activation and pass that value to next connected weights.
