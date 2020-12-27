from NodeGraph import NodeGraph
from TNode import TNode
from Identity import Identity
from Container import Container
from Position import Position
import SharedCounter

class TLayer(Identity, NodeGraph, Container, Position):
    """Pass Number of Tensors"""

    def __init__(self, **kwargs):
        Identity.__init__(self)
        Container.__init__(self)
        NodeGraph.__init__(self)
        Position.__init__(self)

        self.NumTNode = kwargs.get('NumTNode')
        attr = SharedCounter.Counter
        for _ in range(attr, self.NumTNode+1):  # For all tensors
            newTensor = TNode()  # Create a new instance of tensor
            self.add(newTensor)  # add tensor in Container
