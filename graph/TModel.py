from Identity import Identity
from Container import Container
from TLayer import TLayer
from NodeGraph import NodeGraph
from Position import Position
from Data import Data
import SharedCounter

class Model(Identity, Container, Position):

    def __init__(self, **kwargs):
        '''Pass NumLayer=?,NumTensor=?'''

        Identity.__init__(self)
        Container.__init__(self)
        Position.__init__(self)

        self.NumLayers = kwargs.get('NumLayer')
        self.NumTensor = kwargs.get('NumTensor')
        self._Data = None
        # Model Initilization by private functions
        self.__InitLayers()
        self.__ConnectLayers()
        self.__ConnectTensors()
        self.__ConnectWeightsOfTensor()


    def __InitLayers(self):
        '''Initilize all layers and individual tensors from layer'''

        flag = False

        if isinstance(self.NumTensor, (list, set, dict)):
            flag = True

        # Add all layers
        attr = SharedCounter.Counter
        for index in range(attr, self.NumLayers+1):  # for all layers of model
            # Create a new instance of layer
            newlayer = None
            if flag:
                # if an array of number of tensors in each layer is provided
                newlayer = TLayer(NumTNode=self.NumTensor[index-1])
            else:
                # if all layers will have same number of tensors
                newlayer = TLayer(NumTNode=self.NumTensor)
            self.add(newlayer)  # add layer to Container

    def __ConnectLayers(self):
        # Connect individual layers togather,
        newlayer = None
        nextlayer = None
        previouslayer = None
        attr = SharedCounter.Counter
        count = attr
        keys = list(self.Container)
        for index in self.Container:  # for all layers of model
            newlayer = self.Container[index]
            if count == self.NumLayers:
                # Connect last layer to a null layer as next layer
                nextlayer = None
                # set layer as last and first layer
                newlayer.set_PositionAsEnd()
            else:
                # Connect layer to next layer as next layer
                nextlayer = self.Container[keys[keys.index(index)+1]]
                nextlayer.set_PositionAsHidden()

            if count == attr:
                # Connect first layer to last layer as previous layer
                previouslayer = None
            else:
                # Connect layer to previous layer as previous layer
                previouslayer = self.Container[keys[keys.index(index)-1]]
            if isinstance(newlayer, NodeGraph):
                # Layer is instance of graph
                if nextlayer is not None:
                    newlayer.N_connectNode(nextlayer)
                if previouslayer is not None:
                    newlayer.P_connectNode(previouslayer)
            count += 1

    def __ConnectTensors(self):
        attr = SharedCounter.Counter
        # Connect Tensors of a layer to all tensors of another layer
        count = attr
        keys = list(self.Container)
        for index in self.Container:  # for all layers of model
            newlayer = self.Container[index]
            if count == self.NumLayers:
                # Connect last layer to a null layer as next layer
                nextlayer = None
                # set layer as last and first layer
                newlayer.set_PositionAsEnd()
            else:
                # Connect layer to next layer as next layer
                nextlayer = self.Container[keys[keys.index(index)+1]]
                nextlayer.set_PositionAsHidden()

            if count == attr:
                # Connect first layer to last layer as previous layer
                previouslayer = None
            else:
                # Connect layer to previous layer as previous layer
                previouslayer = self.Container[keys[keys.index(index)-1]]
            flag = True
            container_length = len(newlayer.Container)
            inner_container = newlayer.Container
            keys_t = list(inner_container)

            for tnode in inner_container:

                if nextlayer is not None:
                    inner_container[tnode].N_connectNodes(nextlayer.Container)
                if previouslayer is not None:
                    inner_container[tnode].P_connectNodes(
                        previouslayer.Container)

                # set current tensor node position as hidden position if it is not first and last
                if flag:  # start tensor
                    inner_container[tnode].set_PositionAsStart()
                    flag = False
                elif not flag:
                    inner_container[tnode].set_PositionAsHidden()
                if keys_t.index(tnode) == container_length:
                    inner_container[tnode].set_PositionAsEnd()

            count += 1

    def __ConnectWeightsOfTensor(self):
        attr = SharedCounter.Counter
        count = attr
        keys = list(self.Container)
        for index in self.Container:  # for all layers of model
            newlayer = self.Container[index]
            if count == self.NumLayers:
                # Connect last layer to a null layer as next layer
                nextlayer = None
                # set layer as last and first layer
                newlayer.set_PositionAsEnd()
            else:
                # Connect layer to next layer as next layer
                nextlayer = self.Container[keys[keys.index(index)+1]]
                nextlayer.set_PositionAsHidden()

            if count == attr:
                # Connect first layer to last layer as previous layer
                previouslayer = None
            else:
                # Connect layer to previous layer as previous layer
                previouslayer = self.Container[keys[keys.index(index)-1]]

            inner_container = newlayer.Container
            # assign each tensor to all other tensors via weights
            for tnode in inner_container:
                if nextlayer is not None:
                    for _tnode in nextlayer.Container:
                        wt_instance = inner_container[tnode]._getWtInstance(
                            nextlayer.Container[_tnode])
                        inner_container[tnode]._add_N_connectedWeight(
                            wt_instance)
             # assign each tensor to all other tensors via weights
            if previouslayer is not None:
                for _tnode in previouslayer.Container:
                    for tnode in inner_container:
                        wt_instance = previouslayer.Container[_tnode].get_Next_Connected_Wt(tnode)
                        inner_container[tnode]._add_P_connectedWeight(
                            wt_instance, _tnode)
            count += 1

    def setData(self, xtrain, ytrain, xtest, ytest, op='default'):
        self._Data = Data(op,xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest)
         
    def get_AllTNodes(self):
        all_nodes = dict()
        for index in self.Container:  # for all layers of model
            layer = self.Container[index]
            inner_container = layer.Container
            for tnode in inner_container:
                all_nodes[tnode] = inner_container[tnode]

        return all_nodes

    def print(self):
        print(str(self.get_Id()))
