import typing
from NodeGraph import NodeGraph
'''All classes in this file are inherited by each tensor instance (TNode)'''
import hashlib
import random
import SharedCounter 
import numpy as np
import random
from Identity import Identity
class EncodeData():
    def __init__(self, val, option="md5"):
        if option == "md5":
            op = ""
            if isinstance(val, (list,np.ndarray)):
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

class Properties(object):
    def __init__(self):
        self.ActivationVal = None
        self.Loss = dict()
        self.ActivationVal_Storage = dict()
        
    def get_Loss(self):
        return self.Loss

    def set_ActivationVal(self, value):
        self.ActivationVal = value
         
    def get_ActivationVal(self):
        return self.ActivationVal
    
    #for back prop   
    def set_ActivationStorage(self, value, pass_counter):
        #self.set_ActivationVal(value)
        if isinstance(value, np.ndarray):
            value = value.sum() 
        self.ActivationVal_Storage[pass_counter] = value

    def get_ActivationStorage(self, pass_counter):
        return self.ActivationVal_Storage[pass_counter]

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
        if isinstance(self._Activation, (list,np.ndarray)):    
            val = []
            for i in range(0, len(self._Activation)):
                if isinstance(self._Activation[i], str):
                    hash_obj = hashlib.md5(self._Activation.encode())
                    val.append(float.fromhex(hash_obj.hexdigest()))
                else:
                    return self._Activation
            return val
        raise Exception("type not supported.")


    def set_Input(self, _Input, Normalize=None):
        if Normalize is None:
            if isinstance(_Input, list):
                _Input = np.array(_Input)
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


class Weight(Properties,Identity):
    '''Each tensor will have connected Tensors from other layer 
    and each of those tensors will have weight connected to current tensor'''

    def __init__(self, TNode=None):
        self.__NodeWeight = SharedCounter.WEIGHT_START
        self.__NodeInput = None
        self.__Bais = random.uniform( SharedCounter.INITIAL_BAIS, SharedCounter.alpha)
        self.TNode = TNode
        self.LocalMax_Index = 0
        self.LocalMin_Index = 0
        self.LocalMax_Activation = 0
        self.LocalMin_Activation = 1
        Properties.__init__(self)
        Identity.__init__(self)

    def get_NodeWeight(self):
        return self.__NodeWeight

    def set_NodeWeight(self, NodeWeight: float):
        self.__NodeWeight = NodeWeight
        self.__Bais = SharedCounter.INITIAL_BAIS

    def resize_NodeWeight(self, size):
        self.__NodeWeight = None
        self.__Bais = []
        self.__NodeWeight = []
        for _ in range(0, size):
            self.__NodeWeight.append(random.uniform(SharedCounter.WEIGHT_START, SharedCounter.WEIGHT_END))
            self.__Bais.append(random.uniform(SharedCounter.WEIGHT_START, SharedCounter.WEIGHT_END))

    def get_NodeInput(self):
        if isinstance( self.__NodeInput, float):
           # print(str(self.__NodeInput))
            return  self.__NodeInput
        if isinstance(self.__NodeInput, str):
            hash_obj = hashlib.md5(self.__NodeInput.encode())
           # print(str(float.fromhex(hash_obj.hexdigest())))
            return float.fromhex(hash_obj.hexdigest())
        if isinstance(self.__NodeInput, (list,np.ndarray)):    
            return self.__NodeInput
            
        raise Exception("type not supported.")
    def hypothesis(self, in_, wt_):
        return np.multiply(in_ , wt_)
    def set_NodeInput(self, NodeInput):

        self.__NodeInput = NodeInput.sum()
        
        # self.LocalMax_Index = 0
        # self.LocalMin_Index = 0
        # self.LocalMax_Activation = 0
        # self.LocalMin_Activation = 1
        # for _ in range(0, len(self.__NodeInput)):
        #     if isinstance(self.__NodeWeight, (list, np.ndarray)):
        #         x__ = self.__NodeWeight[_] * self.__NodeInput[_]
        #         if  x__ > self.LocalMax_Activation:
        #             self.LocalMax_Index = _
        #             self.LocalMax_Activation = x__
        #         if  x__ < self.LocalMin_Activation:
        #             self.LocalMin_Activation = x__
        #             self.LocalMin_Index = _
                
        #     else:
        #         x__ = self.__NodeWeight * self.__NodeInput[_]
        #         if x__ > self.LocalMax_Activation:
        #             self.LocalMax_Index = _
        #             self.LocalMax_Activation = x__
        #         if  x__ < self.LocalMin_Activation:
        #             self.LocalMin_Activation = x__
        #             self.LocalMin_Index = _

    def set_NodeBais(self, NodeBais:float):
        self.__Bais = NodeBais
    def get_NodeBais(self):
        return self.__Bais

    def _weight_change_call(self,pass_counter):
        loss = self.Loss[pass_counter]
        activation = self.ActivationVal_Storage[pass_counter]
        return (loss, activation)
    
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
