import copy

class SaveTensors(object):
    def __init__(self, allTNodes=dict()):
        self.Correct_Tensors = []
        self.InCorrect_Tensors = []
        self.InCorrect_Tensors_Max = set()
        self.InCorrect_Tensors_Min = set()
        self.All_Tensors = allTNodes
        self.last_layer = []
    
    def set_last_nodes(self):
        for t_id in self.All_Tensors:
            if self.All_Tensors[t_id].islastLayerNode() == True:
                self.last_layer.append(t_id)

    def add_tensor(self, Tensor, flag=False, max=False, min=False):
        
        Tensor_id = Tensor.get_Id()
        if flag:
            self.Correct_Tensors.append(Tensor_id)
        else:
            self.InCorrect_Tensors.append(Tensor_id)
        if max ==True:
            self.InCorrect_Tensors_Max.add(Tensor_id)
        
        if min == True:
            self.InCorrect_Tensors_Min.add(Tensor_id)
    def get_all_max_tensors(self):
        tnodes = dict()
        for _ in self.InCorrect_Tensors_Max:
            tnodes[_] = self.All_Tensors[_]
        return tnodes
    def get_all_min_tensors(self):
        tnodes = dict()
        for _ in self.InCorrect_Tensors_Min:
            tnodes[_] = self.All_Tensors[_]
        return tnodes

    def get_all_correct_tensors(self):
        return self.Correct_Tensors

    def get_all_incorrect_tensors(self):
        return self.InCorrect_Tensors

    def resetArrays(self):
        self.Correct_Tensors.clear()
        self.InCorrect_Tensors.clear()

    def reset(self):
        self.InCorrect_Tensors_Max.clear()
        self.InCorrect_Tensors_Min.clear()



