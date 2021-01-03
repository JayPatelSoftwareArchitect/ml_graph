import math
class LogisticRegression(object):
    def __init__(self):
        self.e = math.exp

    def _calculate(self, b0, b1, x):
        #b0 is bais, b1 is the weight x is the associative input value.
        #returns y , output
        if isinstance(x, list):
            for i in range(0, len(x)):
                y += self.e(b0 + (b1 * self._normalize(x[i]))) / (1 + self.e(b0 + (b1 * self._normalize(x[i]))))
            return y
        else:
            y = self.e(b0 + (b1 * self._normalize(x))) / (1 + self.e(b0 + (b1 * self._normalize(x))))
            return y
    
    def _normalize(self, value):
        normal = None
        if isinstance(value, list):
            normal = []
            for i in range(0, len(value)):
                normal.append((1 / (1 + self.e(0-value[i]))))
        else:
            normal = 1 / (1 + self.e(0-value))
        return normal