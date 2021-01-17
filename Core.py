from GraphImport import *
from Training import Training
from Optimizer import Callback, Random
import numpy as np
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample_x = []
sample_y = []


model = Model(NumLayer=4, NumTensor=[32, 10, 10, 10])
model.setData(x_train[:500], y_train[:500], x_test[:50], y_test[:50],op='md5')
#passing in tensorflow, a encoded dataset
# x_t = np.array(model._Data.xtrain)
# x_ = np.array(model._Data.xtest)
# y_t = model._Data.ytrain
# y_ = model._Data.ytest
# model_tf = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(32,)),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(10)
# ])
# model_tf.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model_tf.fit(x_t, y_t, epochs=1000, batch_size=10)
# test_loss, test_acc = model_tf.evaluate(x_,  y_ , verbose=2)

# print('\nTest accuracy:', test_acc)

training = Training(model, LogisticRegression)

#training.setActivationFunction(LogisticRegression)
training.setCallBack(Callback)
training.fit(optimum_pass=5)

# model.print()
#For each tensor take all previous connected weights and add them up add bais apply activation and pass that value to next connected weights.
