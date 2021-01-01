from NodeGraph import NodeGraph
from Identity import Identity
from Position import Position
from Weights import WeightDict, _Activation, _Bais, _ActivationFn, _CallFn
import typing
import SharedCounter


class TNode(Identity, NodeGraph, Position, WeightDict, _Activation, _Bais, _ActivationFn, _CallFn):
    '''Tensor class that has been deeply inherited'''

    def __init__(self):

        Identity.__init__(self)
        NodeGraph.__init__(self)
        Position.__init__(self)
        WeightDict.__init__(self)
        _Activation.__init__(self, SharedCounter.INITIAL_ACTIVATION)
        _Bais.__init__(self, SharedCounter.INITIAL_BAIS)
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
        val = 0.0
        for wtNode in self.P_ConnectedWt:
            val += self.P_ConnectedWt[wtNode].get_NodeInput() * self.P_ConnectedWt[wtNode].get_NodeWeight()
        self.set_Input(val)

    def _setWeightInputNext(self, _input):
        val = (self.get_ActivationFn())(_input + self.get_Bais())
        #Update node activation value
        self.set_Input(val)
        for wtNode in self.N_ConnectedWt:
            self.N_ConnectedWt[wtNode].set_NodeInput(val)            


