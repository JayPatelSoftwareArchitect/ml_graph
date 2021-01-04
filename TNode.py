from NodeGraph import NodeGraph
from Identity import Identity
from Position import Position
from Internals import WeightDict, _Activation, _Bais, _ActivationFn, _CallFn
from LogisticRegression import LogisticRegression
import typing
import SharedCounter


class TNode(Identity, NodeGraph, Position, WeightDict, _Activation, _Bais, _ActivationFn, _CallFn):
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
            for wtNode in self.P_ConnectedWt:
                val.append( self._ActivationFn._calculate(self.get_Bais(), self.P_ConnectedWt[wtNode].get_NodeWeight(), self.P_ConnectedWt[wtNode].get_NodeInput()) )
            self.set_Input(val)
        else:
            raise Exception("Only logistic regression is supported. Please set ascivationfunction. ")

    def _setWeightInputNext(self, _input):
        #Update node activation value for each next connected weight instances of current node. 
        for wtNode in self.N_ConnectedWt:
            self.N_ConnectedWt[wtNode].set_NodeInput(_input)            


    @staticmethod
    def _util_activation(tensor):
        '''A utility function for a tensor object, that returns calculated total activation'''
        if isinstance(tensor._Activation, list):
            total_activation = 0.0
            for i in range(0, len(tensor._Activation)):
                total_activation += tensor._Activation[i]
            return total_activation
        else:
            return tensor._Activation

    def _compare_tensor(self, tensor_node):
        '''If current tensor's activation is higher then passed tensor return true else false'''
        return TNode._util_activation(self) > TNode._util_activation(tensor_node)

    