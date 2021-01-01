import sys
import math


class Activation():
    def custom(value):
        value_real = complex(value)
        c = math.log(abs(value_real.real))

        z = math.e ** abs(value_real.imag)

        return complex(c * z).real

    def identity(value):
        return value

    def bainarystep(value):
        if value < 0:
            return 0

        return 1

    def softstep(value):
        return (1 / (1 + math.e ** (0-value)))

    def tanh(value):
        return math.tanh(value)

    def relu(value):
        # if value <= 0:
        #     return 0

        return value

    def softplus(value):
        return math.log(1 + (math.e ** value))

    def arctan(value):
        return (1 / math.tan(value))

    def softSign(value):
        return (value / (1+abs(value)))

    def bentIdentity(value):
        return ((math.sqrt((value*value)+1) - 1) / 2)+value

    def sigmoid(value):
        # x / 1 + e ^ -x
        tempV = 1 + math.e ** (0-value)
        return 1 / tempV

    def sign(value):
        return math.sin(value)

    def gaussian(value):
        tempValueSqure = (value * value)  # -x squere
        E = math.e
        return 1/(E ** tempValueSqure)

    def SQRBF(value):
        if abs(value) <= 1:
            return (1-(value*value)/2)  # 1-xsqr/2
        elif 1 <= abs(value) and abs(value) <= 2:
            return ((2-abs(value))*(2-abs(value)))/2  # (2-|x|sqr / 2)
        elif abs(value) >= 2:
            return 0

    def signC(value):
        if value == 0:
            return 1
        # if x = 0, 1 else sin(x)/x
        return math.sin(value)/value
