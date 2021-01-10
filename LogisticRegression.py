import math
import numpy as np
class LogisticRegression(object):
    def __init__(self):
        self.e = math.exp

    def _calculate(self, b0, b1, x, y):
        #b0 is bais, b1 is the weight x is the associative input value.
        #returns y , output
        if isinstance(x, (list, np.ndarray)):
            
            for i in range(0, len(x)):
                if isinstance(b1, (list, np.ndarray)):
                    y += b0 + (b1[i] * x[i]) / (1 + b0 + (b1[i] * x[i]))
                else:
                    y += b0 + (b1 * x[i]) / (1 + b0 + (b1 * x[i]))
              
            return self._normalize(y)
        else:
            y += b0 + (b1 * x) / (1 + b0 + (b1 * x))
            return self._normalize(y)
    
    def _normalize(self, value):
        normal = None
        # if isinstance(value, list):
        #     normal = []
        #     for i in range(0, len(value)):
        #         normal.append(self.e(value[i]))
        # else:
        normal = 1/(1 + self.e(0-value))
        
        return normal