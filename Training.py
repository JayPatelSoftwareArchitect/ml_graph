from TModel import Model
from Utility import Utility
from functools import cmp_to_key
import copy

class Callback(object):
    def __init__(self):
        self.TNode = []
        self.TNodeCounter = 0
        self.TNodeSelect = 0
    def addNode(self, node):
        self.TNode.append(node)

    def remove(self):
        self.TNode.pop()
    
    def set_EachOp(self, select):
        self.TNodeSelect = select

    def callIter(self):
        for _ in range(0,self.TNodeSelect):
            tensor = self.TNode[self.TNodeCounter]
            tensor._CallFn()
            self.TNodeCounter+=1
            if self.TNodeCounter > len(self.TNode):
                self.TNodeCounter = 0

class SaveTensors(object):
    def __init__(self):
        self.Correct_Tensors = []
        self.InCorrect_Tensors = []

    def add_tensor(self, Tensor, flag=False):
        if flag:
            self.Correct_Tensors.append(copy.copy(Tensor))
        else:
            self.InCorrect_Tensors.append(copy.copy(Tensor))
    def get_all_correct_tensors(self):
        return self.Correct_Tensors

    def get_all_incorrect_tensors(self):
        return self.InCorrect_Tensors
    
class Training(SaveTensors):
    def __init__(self, model):
        if isinstance(model, Model):
            self.Model = model
            self.Output_Options = []
            self.Input_Options = []
            self.Callback = None
            self.LastLayerId = None
            self.FirstLayerId = None
            SaveTensors.__init__(self)
        else :
            raise RuntimeError("passed model is not of type Model")

    def sample(self, Input_Options=[], Output_Options=[]):
        #add sample input values and sample output values (all possible outputs).
        self.Input_Options = Input_Options
        self.Output_Options = Output_Options

    def setActivationFunction(self, fn):
        #pass a function that activates the tensor, it should take in a float value and should return a calculated value
        for layer_id in self.Model.Container:
            layer = self.Model.Container[layer_id]
            for tensor_id in layer.Container:
                tensor = layer.Container[tensor_id]
                tensor.set_ActivationFn(fn)

    def setCallFunction(self, callfn):
        for layer_id in self.Model.Container:
            layer = self.Model.Container[layer_id]
            for tensor_id in layer.Container:
                tensor = layer.Container[tensor_id]
                tensor.set_CallFn(callfn)
        # simple algo which will select only one tensor of the model on each training example
        # It will select weights such that it can activate that tensor if current activation value 
        # is less then threshold value.
    def setCallBack(self, number_of_selection=1,default=0):
        #number_of_selection is an int of howmany tensors will be adjusted be _CallFn per each callIter() default=0 will add all tensors of model
        if default == 0:
            self.Callback = Callback()
            #add all tensors
            for layer_id in self.Model.Container:
                layer = self.Model.Container[layer_id]
                for tensor_id in layer.Container:    
                    tensor = layer.Container[tensor_id]
                    self.Callback.addNode(tensor)
            self.Callback.set_EachOp(number_of_selection)

    def setInput(self, input_=[]):
        first_l = True
        lid = 0
        for layer_id in self.Model.Container:
            lid = layer_id
            layer = self.Model.Container[layer_id]
            TASKS_INPUT = []
            if first_l is True:
                first_l = False
                self.FirstLayerId = lid
                input_c = 0
                for tensor_id in layer.Container:
                    tensor =  layer.Container[tensor_id]
                    #tensor._calculateActivation(input_[input_c])
                    TASKS_INPUT.append((tensor._calculateActivation,input_[input_c]))
                    #print(str(input_[input_c]))
                    input_c+=1
            else:    
                for tensor_id in layer.Container:
                    tensor =  layer.Container[tensor_id]
                    #tensor._calculateActivation()                 
                    TASKS_INPUT.append(tensor._calculateActivation)
            #execute each task parallely for all tensors of first layer.
            try:
                layer.callOnEach(4 , TASKS_INPUT)
            except:
                print("exception")
        self.LastLayerId = lid

    def pass_(self,input_=[], outputs=[]):
        #set input values in first layer tensors
        self.setInput(input_)
        val = []
        index = 0
        for tensor_id in self.Model.Container[self.LastLayerId].Container:
            tensor = self.Model.Container[self.LastLayerId].Container[tensor_id]
            val.append((tensor, index))
            index+=1

        #sorted(val,key=cmp_to_key(lambda t1,t2: Utility.compare(t1,t2)))
        max_ = None
        for i in range(0, index):
            if max_ is None:
                max_ = val[i] 
            elif max_[0]._Activation < val[i][0]._Activation:
                max_ = val[i]
        print(str(max_[0]._Activation))
        print("Selected tensor is : "+str(max_[1]) + " Actual: " + str(outputs))
        if max_[1] != outputs:
            self.add_tensor(max_[0]) #incorrect output from model
        else:
            self.add_tensor(max_[0], True) #correct output from model
        print("\n Next \n")

  

    def fit(self, inputs=[], outputs=[]):
        for index in range(0, len(inputs)) :
            inp_ = inputs[index]
            op_ = outputs[index]
            self.pass_(inp_, op_)
