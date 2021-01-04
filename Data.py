import hashlib
import numpy as np

class Data(object):
    '''A class that will be used to wrap input to the model'''
    def __init__(self,op='default', xtrain=None, ytrain=None, xtest=None, ytest=None, encode="all",saperate=" "):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest

        #encoded.
        self.extrain = None
        self.eytrain = None
        self.extest = None
        self.eytest = None
        
        self.accuracy = 0.0
        if op == 'md5':
            self._stringInterop(op,encode, saperate)

    def _stringInteropHelper(self, arr=[], saperate=" "):
        temp = ""
        if isinstance(arr, (list, np.ndarray) ):
            for i in range(0, len(arr)):
                temp += self._stringInteropHelper(saperate, arr[i])    
        elif isinstance(arr, tuple):
            temp += arr[0]
            temp += saperate
            temp += arr[1]
            temp += saperate    
        elif isinstance(arr, (int, float) ):
            temp = str(arr)
        elif isinstance(arr, str):
            temp = arr
        return temp

    def _stringInterop(self, op='md5', encode='all', saperate=" "):
        '''A functionality that takes in array of numbers and encodes it as a string saperated by space. Then the hex string becomes a single input to model. This is experimental.'''
        if self.xtrain is None or self.ytrain is None or self.xtest is None or self.ytest is None:
            raise Exception("Initilize data() with arrays of input output train, test data")
        if encode != 'all' and encode != 'train' and encode != 'test':
            raise Exception("Only 3 options, all will encode train and test, or train or test")

        if op == 'md5':
            if encode == 'all':
                self.extrain = []
                self.eytrain = self.ytrain
                self.extest = []
                self.eytest = self.ytest
                for i in range(0, len(self.xtrain)):
                    str_ = self._stringInteropHelper(self.xtrain[i], saperate=saperate)
                    hex_ = hashlib.md5(str_.encode())
                    digest_ = hex_.hexdigest()
                    float_arr = []
                    #digest will be 32 bit char arr.
                    for j in range(0, 32):
                        float_arr.append(float.fromhex(digest_[j]))
                    self.extrain.append(float_arr)

                for i in range(0, len(self.xtest)):
                    str_ = self._stringInteropHelper(self.xtest[i], saperate=saperate)
                    hex_ = hashlib.md5(str_.encode())
                    digest_ = hex_.hexdigest()
                    #digest will be 32 bit char arr.
                    float_arr = []
                    for j in range(0, 32):
                        float_arr.append(float.fromhex(digest_[j]))
                    self.extest.append(float_arr)

        else:
            raise Exception("Only md5 is supported. will add sha and prime..")