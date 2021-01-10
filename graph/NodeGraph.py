from Identity import Identity
# all shared methods and properties for individual node
# (a tensor or a container(lot of tensors) or a multicontainer) will be writen


class NodeGraph(object):
    '''For connecting layer to another layer'''

    def __init__(self):
        # Next connected and Previous Connected nodes
        self.N_Connected = dict()
        self.P_Connected = dict()

    def N_connectNode(self, Instance):
        '''For connecting tensor to another next layer tensor'''
        self.N_Connected[Instance.get_Id()] = Instance

    def P_connectNode(self, Instance):
        '''For connecting tensor to another previous layer tensor'''
        self.P_Connected[Instance.get_Id()] = Instance

    def N_connectNodes(self, Instance):
        '''For connecting Tensors of a layer to all tensors of another layer'''

        for ins in Instance:
            self.N_Connected[Instance[ins].get_Id()] = Instance[ins]

    def P_connectNodes(self, Instance):
        '''For connecting Tensors of a layer to all tensors of another layer'''

        for ins in Instance:
            self.P_Connected[Instance[ins].get_Id()] = Instance[ins]
