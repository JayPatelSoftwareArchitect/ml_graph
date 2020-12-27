import typing
from NodeGraph import NodeGraph
'''All classes in this file are inherited by each tensor instance (TNode)'''


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

    def __init__(self, _Input: float):
        self._Activation = _Input

    def get_Input(self) -> float:
        return self._Activation

    def set_Input(self, _Input: float) -> float:
        self._Activation = _Input

class _ActivationFn(object):
    '''A function wraper (for assigning activation function to individual tensor)'''

    def __init__(self):
        self._ActivationFn = None

    def get_ActivationFn(self):
        return self._ActivationFn

    def set_ActivationFn(self, _ActivationFn):
        self._ActivationFn = _ActivationFn


class Weight(object):
    '''Each tensor will have connected Tensors from other layer 
    and each of those tensors will have weight connected to current tensor'''

    def __init__(self, TNode=None):
        self.__NodeWeight = 0
        self.__NodeInput = 0
        self.TNode = TNode

    def get_NodeWeight(self):
        return self.__NodeWeight

    def set_NodeWeight(self, NodeWeight: float):
        self.__NodeWeight = NodeWeight

    def get_NodeInput(self):
        return self.__NodeInput

    def set_NodeInput(self, NodeInput: float):
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

    def _add_P_connectedWeight(self, Weight: Weight):
        self.P_ConnectedWt[Weight.TNode.get_Id()] = Weight
        self.NLength += 1

    def get_Prev_Connected_Wt(self, TNodeId: int) -> float:
        if TNodeId in self.P_ConnectedWt:
            return self.P_ConnectedWt[TNodeId]

    def get_Next_Connected_Wt(self, TNodeId: int) -> float:
        if TNodeId in self.N_ConnectedWt:
            return self.N_ConnectedWt[TNodeId]
