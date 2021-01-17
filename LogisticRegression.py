import math
import numpy as np
class LogisticRegression(object):
    def __init__(self):
        self.e = math.exp
        self.threshold = 0.5

    def _calculate(self, b0, b1, x):
        #hypothesis function
        #b0 is bais, b1 is the weight x is the associative input value.
        #returns y , output
        y = 0.0
        if isinstance(x, (list, np.ndarray)):
            if isinstance(b1, (list, np.ndarray)):            
                y = np.add(b0, np.multiply(b1,x)).sum() / len(x)
            else:
                for i in range(0, len(x)):
                    y += b0[i] + (b1[i] * x[i]) / (1 + b0[i] + (b1[i] * x[i]))
                else:
                    y += b0 + (b1 * x[i]) / (1 + b0 + (b1 * x[i]))
              
            return self._isactivated(y)
        else:
            y += b0 + (b1 * x) / (1 + b0 + (b1 * x))
            return self._isactivated(y)
    
    def _isactivated(self, value):
        
        # if isinstance(value, list):
        #     normal = []
        #     for i in range(0, len(value)):
        #         normal.append(self.e(value[i]))
        # else:
        # # print(str(value) + "\n")
        # if value >= self.threshold:
        #     normal = 1/(1 + self.e(0-abs(value)))
        # else:
        #     normal = 0.0
        if value > self.threshold:
            return value
        else:
            return 0.0