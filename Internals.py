import typing
from NodeGraph import NodeGraph
'''All classes in this file are inherited by each tensor instance (TNode)'''
import hashlib
import random
import SharedCounter 

class EncodeData():
    def __init__(self, val, option="md5"):
        if option == "md5":
            op = ""
            if isinstance(val, list):
                for i in range(0, len(val)):
                    hash_obj = hashlib.md5(val[i].encode())
                    op.append(hash_obj.hexdigest())
            self.getHash = op

class _Bais(object):
    '''Each Tensor will have an instance of Bais'''

    def __init__(self, _Bais: float):
        self._Bais = _Bais

    def get_Bais(self) -> float:
        return self._Bais

    def set_Bais(self, _Bais: float) -> float:
        self._Bais = _Bais


class _Activation(object):
    '''A input (floating value comming in to model or from other tensors)'''

    def __init__(self, _Input):
        self._Activation = _Input

    def get_Input(self):
        if isinstance(self._Activation, float):
            #print(str(self._Activation))
            return  self._Activation
        if isinstance(self._Activation, str):
            hash_obj = hashlib.md5(self._Activation.encode())
            #print(str(float.fromhex(hash_obj.hexdigest())))
            return float.fromhex(hash_obj.hexdigest())
        if isinstance(self._Activation, list):    
            val = []
            for i in range(0, len(self._Activation)):
                if isinstance(self._Activation[i], str):
                    hash_obj = hashlib.md5(self._Activation.encode())
                    val.append(float.fromhex(hash_obj.hexdigest()))
                else:
                    val.append(self._Activation[i])
            return val
        raise Exception("type not supported.")


    def set_Input(self, _Input, Normalize=None):
        if Normalize is None:
            self._Activation = _Input
        else:
            self._Activation = Normalize(_Input)
class _ActivationFn(object):
    '''A function wraper (for assigning activation function to individual tensor)'''

    def __init__(self):
        self._ActivationFn = None

    def get_ActivationFn(self):
        return self._ActivationFn

    def set_ActivationFn(self, _ActivationFn):
        self._ActivationFn = _ActivationFn()

class _CallFn(object):
    '''A function wraper (for assigning call function to individual tensor) which will adjust weights of next connected tensors'''

    def __init__(self):
        self._CallFn = None

    def get_CallFn(self):
        return self._CallFn

    def set_CallFn(self, _CallFn):
        self._CallFn = _CallFn


class Weight(object):
    '''Each tensor will have connected Tensors from other layer 
    and each of those tensors will have weight connected to current tensor'''

    def __init__(self, TNode=None):
        self.__NodeWeight = random.uniform(SharedCounter.WEIGHT_START , SharedCounter.WEIGHT_END)
        self.__NodeInput = None
        self.TNode = TNode

    def get_NodeWeight(self):
        return self.__NodeWeight

    def set_NodeWeight(self, NodeWeight: float):
        self.__NodeWeight = NodeWeight

    def get_NodeInput(self):
        if isinstance( self.__NodeInput, float):
           # print(str(self.__NodeInput))
            return  self.__NodeInput
        if isinstance(self.__NodeInput, str):
            hash_obj = hashlib.md5(self.__NodeInput.encode())
           # print(str(float.fromhex(hash_obj.hexdigest())))
            return float.fromhex(hash_obj.hexdigest())
        if isinstance(self.__NodeInput, list):    
            val = 0.0
            for i in range(0, len(self.__NodeInput)):
                if isinstance(self.__NodeInput[i], str):
                    hash_obj = hashlib.md5(self.__NodeInput.encode())
                    val += float.fromhex(hash_obj.hexdigest())
                else:
                    val += self.__NodeInput[i]
            return val
        raise Exception("type not supported.")

    def set_NodeInput(self, NodeInput):
        self.__NodeInput = NodeInput

class WeightDict(object):
    '''Each tensor will have a Dictionary that holds Next connected weights
    and Previous connected Weights For Store of 
    Weighted Nodes (Weight) instances'''

    def __init__(self):
        self.PLength = 0
        self.NLength = 0
        # Next connected and Previous Connected node weights
        self.N_ConnectedWt = dict()
        self.P_ConnectedWt = dict()

    def _getWtInstance(self, TNode=None):
        return Weight(TNode=TNode)

    def _add_N_connectedWeight(self, Weight: Weight):
        self.N_ConnectedWt[Weight.TNode.get_Id()] = Weight
        self.PLength += 1

    def _add_P_connectedWeight(self, Weight: Weight, prevId: int):
        #print(str(prevId))
        self.P_ConnectedWt[prevId] = Weight
        self.NLength += 1

    def get_Prev_Connected_Wt(self, TNodeId: int):
        if TNodeId in self.P_ConnectedWt:
            return self.P_ConnectedWt[TNodeId]

    def get_Next_Connected_Wt(self, TNodeId: int):
        if TNodeId in self.N_ConnectedWt:
            return self.N_ConnectedWt[TNodeId]
