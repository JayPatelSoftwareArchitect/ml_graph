import copy

class SaveTensors(object):
    def __init__(self):
        self.Correct_Tensors = []
        self.InCorrect_Tensors = []
        self.InCorrect_Tensors_Max = dict()
        self.InCorrect_Tensors_Min = dict()
    def add_tensor(self, Tensor, flag=False, max=False, min=False):
        if flag:
            self.Correct_Tensors.append(Tensor)
        else:
            self.InCorrect_Tensors.append(Tensor)
        if max ==True:
            self.InCorrect_Tensors_Max[Tensor.get_Id()] = Tensor
        
        if min == True:
            self.InCorrect_Tensors_Min[Tensor.get_Id()] = Tensor

    def get_all_correct_tensors(self):
        return self.Correct_Tensors

    def get_all_incorrect_tensors(self):
        return self.InCorrect_Tensors
    
    def reset(self):
        self.InCorrect_Tensors_Max = dict()
        self.InCorrect_Tensors_Min = dict()



