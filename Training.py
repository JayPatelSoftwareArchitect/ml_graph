from TModel import Model
from Utility import Utility
from functools import cmp_to_key
from Data import Data
from TNode import TNode
import copy
import SharedCounter
import random
       
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
    
    def reset(self):
        self.Correct_Tensors = []
        self.InCorrect_Tensors = []
class Training(SaveTensors):
    def __init__(self, model):
        if isinstance(model, Model):
            self.Model = model
            self.Output_Options = []
            self.Input_Options = []
            self.Callback = None
            self.LastLayerId = None
            self.FirstLayerId = None
            self.ParallelExecute = False
            self.previous_accuracy = 0.0
            self.BestModelCopy=model
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
                    
                    #parallel execution or sequntial execution
                    if self.ParallelExecute == False:

                        tensor._calculateActivation(input_[input_c])
                    else:
                        TASKS_INPUT.append((tensor._calculateActivation,input_[input_c]))
                    #print(str(input_[input_c]))
                    input_c+=1
            else:    
                for tensor_id in layer.Container:
                    tensor =  layer.Container[tensor_id]
                    if self.ParallelExecute == False:

                        tensor._calculateActivation()                 
                    else:
                        TASKS_INPUT.append(tensor._calculateActivation)
            
            if self.ParallelExecute == True:
            #execute each task parallely for all tensors of first layer.
                try:
                    layer.callOnEach(4 , TASKS_INPUT)
                except:
                    print("exception")
        self.LastLayerId = lid

    def optimize_randomly(self, max_tensor=None, min_tensor=None, seen_set=None):
        step_start = SharedCounter.RANDOM_STEP
        step_end = SharedCounter.RANDOM_STEP_END
        # will try to reduce weights of max tensors , and increse weights of min_tensor randomly
        if isinstance(max_tensor, TNode):
            if max_tensor.P_ConnectedWt is not None:
                #iterate through all previous weights .
                for wt_ in max_tensor.P_ConnectedWt:
                    weight_instance = max_tensor.P_ConnectedWt[wt_]
                    weight_instance.set_NodeWeight(weight_instance.get_NodeWeight() - random.uniform(step_start, step_end))
     
                    if seen_set.__contains__(weight_instance) is False:
                        seen_set.add(weight_instance)
                        #print(weight_instance.TNode.get_Id())
                        self.optimize_randomly(weight_instance.TNode, seen_set=seen_set)


        if isinstance(min_tensor, TNode):
            if min_tensor.P_ConnectedWt is not None:
                #iterate through all previous weights .
                for wt_ in min_tensor.P_ConnectedWt:
                    weight_instance = min_tensor.P_ConnectedWt[wt_]
                    weight_instance.set_NodeWeight(weight_instance.get_NodeWeight() + random.uniform(step_start, step_end))
                   
                    if seen_set.__contains__(weight_instance) is False:
                        seen_set.add(weight_instance)
                        #print(weight_instance.TNode.get_Id())
                        self.optimize_randomly(weight_instance.TNode, seen_set=seen_set)

    def _pass_helper(self, length, op_tensors, actual):
        max_ = None
        for i in range(0, length):
            if max_ is None:
                max_ = op_tensors[i] 
            elif op_tensors[i][0]._compare_tensor(max_[0]):
                max_ = op_tensors[i]
        
        print(str(TNode._util_activation( max_[0])))
        print("Selected tensor is : "+str(max_[1]) + " Actual: " + str(actual))
        
        if max_[1] != actual:
            seen_set = set()

            if max_[0]._compare_tensor(op_tensors[actual][0]):
                #max [0] is higher then actual.

                self.optimize_randomly(max_[0], op_tensors[actual][0], seen_set)
            else:
                #max [0] is higher then actual.
                self.optimize_randomly(op_tensors[actual][0], max_[0], seen_set)

                
            self.add_tensor(max_[0]) #incorrect output from model
        else:
            self.add_tensor(max_[0], True) #correct output from model
      

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
        self._pass_helper(index,val,outputs)
        
        print("\n Next \n")

    def calculate_accuracy(self):
        '''accuracy = the total correct predictions / total predictions (between 0.0 and 1)'''
        accuracy = len(self.Correct_Tensors) / (len(self.Correct_Tensors) + len(self.InCorrect_Tensors))  
        if self.previous_accuracy > accuracy:
            self.Model = self.BestModelCopy
        elif self.previous_accuracy < accuracy:
            self.BestModelCopy = self.Model
            self.previous_accuracy = accuracy
            
        if accuracy > 0.5:
            return False
        else:
            return True

    def fit(self, inputs=[], outputs=[]):
        if isinstance(self.Model._Data, Data):
            if self.Model._Data.extrain is not None:
                continue_training = True
                count_pass = 1
                while(continue_training):
                    #reset saved tensors
                    self.reset()
                    for index in range(0, len(self.Model._Data.extrain)) :
                        inp_ = self.Model._Data.extrain[index]
                        op_ = self.Model._Data.eytrain[index]
                        self.pass_(inp_, op_)
                    continue_training = self.calculate_accuracy()
                    print("Pass: "+str(count_pass))
                    count_pass += 1