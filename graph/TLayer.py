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
        self.pass_counter = 1

        attr = SharedCounter.Counter
        for _ in range(attr, self.NumTNode+1):  # For all tensors
            newTensor = TNode(layer=self)  # Create a new instance of tensor
            self.add(newTensor)  # add tensor in Container
    
    def reset_activation_storage(self):
        self.pass_counter = 1
      
    def _dynamic_init(self, size):
        '''This will update connected weight instances of tensor with same length as input'''
        
        for tnode_id in self.Container:
            tnode = self.Container[tnode_id]
            for wt in tnode.N_ConnectedWt:
                weightInstance = tnode.N_ConnectedWt[wt]
                weightInstance.resize_NodeWeight(size)
    
    def updatePassInfo(self):
        self.pass_counter += 1

    def set_loss_backwards(self):
        for key in self.Container:
            tnode = self.Container[key]
            total_loss = 0.0
            for wt_ in tnode.N_ConnectedWt:
                weight_instance = tnode.N_ConnectedWt[wt_]
                total_loss += weight_instance.Loss[self.pass_counter]
            tnode.set_Loss(total_loss)
        for prev in self.P_Connected:
            p_layer = self.P_Connected[prev]
            p_layer.set_loss_backwards()
    
    def set_loss_last(self, correct_node, mean_):
        loss_0s = 0
        for key in self.Container:
            tnode = self.Container[key]
            loss_ = mean_ - tnode.get_ActivationVal() 
            # if loss_ == 0.0:
            #     loss_ = correct_node.get_ActivationVal()
            tnode.set_Loss(loss_)
        if loss_0s == len(self.Container):
            for key in self.Container:
                tnode = self.Container[key]
                loss_ = np.random.random_integers(1)
                tnode.set_Loss(loss_)
        #left here, after setting wt instances loss, calculate tnode of loss that connects to wt instances
        for prev in self.P_Connected:
            p_layer = self.P_Connected[prev]
            p_layer.set_loss_backwards()

