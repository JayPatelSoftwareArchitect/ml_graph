from NodeGraph import NodeGraph
from TNode import TNode
from Identity import Identity
from Container import Container
from Position import Position
import SharedCounter
import numpy as np
class TLayer(Identity, NodeGraph, Container, Position):
    """Pass Number of Tensors"""

    def __init__(self, **kwargs):
        Identity.__init__(self)
        Container.__init__(self)
        NodeGraph.__init__(self)
        Position.__init__(self)

        self.NumTNode = kwargs.get('NumTNode')
        #here nodeactivations will be for each pass [tuple(nodeid,nodeactivation)]
        self.NodeActivations = list()
        self.NodeActivations.append(list())
        self.pass_counter = 0

        attr = SharedCounter.Counter
        for _ in range(attr, self.NumTNode+1):  # For all tensors
            newTensor = TNode(layer=self)  # Create a new instance of tensor
            self.add(newTensor)  # add tensor in Container
    def reset_activation_storage(self):
        self.pass_counter = 0
        self.NodeActivations.clear()
        self.NodeActivations.append(list())

    def _dynamic_init(self, size):
        '''This will update connected weight instances of tensor with same length as input'''
        
        for tnode_id in self.Container:
            tnode = self.Container[tnode_id]
            for wt in tnode.N_ConnectedWt:
                weightInstance = tnode.N_ConnectedWt[wt]
                weightInstance.resize_NodeWeight(size)
    
    def updatePassInfo(self):
        self.pass_counter += 1
        self.NodeActivations.append(list())
  