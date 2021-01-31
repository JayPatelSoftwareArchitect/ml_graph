import hashlib
import numpy as np

class Data(object):
    '''A class that will be used to wrap input to the model'''
    def __init__(self,op='default', xtrain=None, ytrain=None, xtest=None, ytest=None):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest

        #encoded.
        
        self.accuracy = 0.0
        if op == 'md5':
            En = Encode(self)
            encoded = En._stringInterop(op,self.xtrain,self.xtest)
            self.xtrain = encoded[0]
            self.xtest = encoded[1] 

        if op == 'flatten':
            xt_ = list()
            yt_ = list()
            En = Encode(self)
            for _ in range(0,len(self.xtrain)):
                xt_.append([])
                instance = self.xtrain[_]
                En._get_Normalized_array(instance, xt_[_])
                xt_[_] = np.asarray(xt_[_])
            for _ in range(0,len(self.xtest)):
                yt_.append([])
                En._get_Normalized_array(self.xtest[_], yt_[_])
                yt_[_] = np.asarray(yt_[_])
            
            self.xtrain =  xt_ 
            self.xtest =  yt_ 
            
class Encode(object):
    def __init__(self, data):
        self = data
    def _stringInteropHelper(self, arr=[], saperate=""):
        temp = ""
        if isinstance(arr, (list, np.ndarray) ):
            for i in range(0, len(arr)):
                temp += self._stringInteropHelper(arr[i],saperate)    
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
    def _get_Normalized_array(self, arr=[], add=[]):
        
        if isinstance(arr, (int, float)):
            add.append(arr)
        else:
            for i in range(0, len(arr)):
                instance = arr[i]
                self._get_Normalized_array(instance, add)
            
    def _stringInterop(self, op='md5',xtrain=[], xtest=[] ):
        
        saperate=""
        '''A functionality that takes in array of numbers and encodes it as a string saperated by space. Then the hex string becomes a single input to model. This is experimental.'''
       
        if op == 'md5':

            extrain = []
            extest = []
            for i in range(0, len(xtrain)):
                str_ = self._stringInteropHelper(xtrain[i], saperate=saperate)
                hex_ = hashlib.md5(str_.encode())
                digest_ = hex_.hexdigest()
                float_arr = []
                    #digest will be 32 bit char arr.
                for j in range(0, 32):
                    float_arr.append(float.fromhex(digest_[j]))
                extrain.append(float_arr)

            for i in range(0, len(xtest)):
                str_ = self._stringInteropHelper(xtest[i], saperate=saperate)
                hex_ = hashlib.md5(str_.encode())
                digest_ = hex_.hexdigest()
                    #digest will be 32 bit char arr.
                float_arr = []
                for j in range(0, 32):
                    float_arr.append(float.fromhex(digest_[j]))
                extest.append(float_arr)

            return (extrain,extest)

        else:
            
            raise Exception("Only md5 is supported. will add sha and prime..")