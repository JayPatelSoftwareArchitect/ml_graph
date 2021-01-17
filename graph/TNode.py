from NodeGraph import NodeGraph
from Identity import Identity
from Position import Position
from Internals import WeightDict, _Activation, _Bais, _ActivationFn, _CallFn, Properties
from LogisticRegression import LogisticRegression
import typing
import numpy as np
import SharedCounter


class TNode(Identity, NodeGraph, Position, WeightDict, _Activation, _Bais, _ActivationFn, _CallFn, Properties) :
    '''Tensor class that has been deeply inherited'''

    def __init__(self):
        #id class
        Identity.__init__(self)
        #A wraped graph instance
        NodeGraph.__init__(self)
        #A position identifier
        Position.__init__(self)
        #A wraped weight dictionary 
        WeightDict.__init__(self)
        #Node input class
        _Activation.__init__(self, SharedCounter.INITIAL_ACTIVATION)
        #Node Bais class
        _Bais.__init__(self, SharedCounter.INITIAL_BAIS)
        #Activation function , set at time of model init. 
        _ActivationFn.__init__(self)
        #properties for each tnode, for different optimizing algorithms.
        Properties.__init__(self)

    def _calculateActivation(self, _first_input=None):
        #first layer node then set input as passed value. Else set input as multiplication of all previous node's activations and weights to current activation value.
        if _first_input is not None:
            self._setActivationFirst(_first_input)
        else :
            self._setActivation()
        #set all next connected weight input with value
        self._setWeightInputNext(self.get_Input())

    def _setActivationFirst(self, _first_input):
        #for first layer nodes the input will be the actual input value.
        self.set_Input(_first_input)


    def _setActivation(self):
        #reset current node Activation value by iterating through all previous weights and it's input. 
        if isinstance(self.get_ActivationFn(), LogisticRegression):
            # logistic regression
            val = []
            #a total activtion from all previous connected weights.
            total_activation = 0.0 

            for wtNode in self.P_ConnectedWt:
                _ = self._ActivationFn._calculate(self.P_ConnectedWt[wtNode].get_NodeBais(), self.P_ConnectedWt[wtNode].get_NodeWeight(), self.P_ConnectedWt[wtNode].get_NodeInput())
                val.append(_) 
                total_activation += _
            #set input as an activated array.
            self.set_Input(val)
            #set total activation
            self.set_ActivationVal(total_activation) 
        else:
            raise Exception("Only logistic regression is supported. Please set ascivationfunction. ")

    def _setWeightInputNext(self, _input):
        #Update node activation value for each next connected weight instances of current node. 
        for wtNode in self.N_ConnectedWt:
            self.N_ConnectedWt[wtNode].set_NodeInput(_input)            


    @staticmethod
    def _util_activation(tensor1, tensor2):
        '''A utility function for a tensor object, that returns calculated total activation'''
        op1_max = 0
        op2_max = 0
        op1_max = tensor1.get_ActivationVal()
        op2_max = tensor2.get_ActivationVal()
      
        return op1_max > op2_max
        
    def _compare_tensor(self, tensor_node):
        '''If current tensor's activation is higher then passed tensor return true else false'''
        return TNode._util_activation(self, tensor_node)

    