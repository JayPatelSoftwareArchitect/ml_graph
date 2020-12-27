from TModel import Model
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample_x = []
for i in range(0, len(x_train)):
    temp_ = []
    for j in range(0, len(x_train[i])):
        for k in range(0, len(x_train[i][j])):
            temp_.append(x_train[i][j][k])
    sample_x.append(temp_)

model = Model(NumLayer=10, NumTensor=[28, 4, 5, 6, 7, 8, 9, 10, 11, 12])
#   model = Model(NumLayer=10, NumTensor=10)

model.print()
#For each tensor take all previous connected weights and add them up add bais apply activation and pass that value to next connected weights.
