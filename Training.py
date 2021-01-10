
from GraphImport import *
from Optimizer import Optimizer
class Training(SaveTensors, Optimizer):
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
            self.current_accuracy = 0.0
            self.counter_ep = 0
            self.epoch = 1
            Optimizer.__init__(self,optimizer='random')
            SaveTensors.__init__(self)
        else :
            raise RuntimeError("passed model is not of type Model")

    def dynamic_init(self):
        '''This function will use single input instance to set type of connected weights, ex if input is an array then each connected edge of tnode will have an array type weight with same length.'''
        if isinstance(self.Model._Data, Data):
            if self.Model._Data.xtrain is not None:
                size = len(self.Model._Data.xtrain[0])
                for layer_id in self.Model.Container: 
                    self.Model.Container[layer_id]._dynamic_init(size)                    

    def setActivationFunction(self, fn):
        #pass a function that activates the tensor, it should take in a float value and should return a calculated value
        for layer_id in self.Model.Container:
            layer = self.Model.Container[layer_id]
            for tensor_id in layer.Container:
                tensor = layer.Container[tensor_id]
                tensor.set_ActivationFn(fn)
    
    def setCallBack(self, Callback,number_of_selection=1,default=0):
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
                for tensor_id in layer.Container:
                    tensor =  layer.Container[tensor_id]
                    
                    #parallel execution or sequntial execution
                    if self.ParallelExecute == False:

                        tensor._calculateActivation(input_)
                    else:
                        TASKS_INPUT.append((tensor._calculateActivation,input_))

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

    def _pass_helper(self, length, op_tensors, actual):
        max_ = None
        correct_one = None
        for i in range(0, length):
            if max_ is None:
                max_ = op_tensors[i] 
            elif op_tensors[i][0]._compare_tensor(max_[0]):
                max_ = op_tensors[i]
            if op_tensors[i][1] == actual:
                correct_one = op_tensors[i][0]
        # print(str(TNode._util_activation( max_[0])))
        #print("Selected tensor is : "+str(max_[1]) + " Actual: " + str(actual))
        self.counter_ep += 1
        if max_[0] != correct_one:
            self.add_tensor(correct_one)
            if max_[0]._compare_tensor(op_tensors[actual][0]):
                #max [0] is higher then actual.
                self.add_tensor(max_[0],max=True) #incorrect output from model
                
            else:
                #max [0] is higher then actual.
                self.add_tensor(max_[0],min=True) #incorrect output from model
                
        else:
            self.add_tensor(max_[0], flag=True) #correct output from model
        if self.counter_ep % self.epoch == 0:
            self.Optimizer.call(self.InCorrect_Tensors_Max, self.InCorrect_Tensors_Min)
            self.reset()

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
        
        # print("\n Next \n")

    def calculate_accuracy(self):
        '''accuracy = the total correct predictions / total predictions (between 0.0 and 1)'''
        self.previous_accuracy = self.current_accuracy
        accuracy = 0.0
        if len(self.Correct_Tensors) > 0:
            accuracy = len(self.Correct_Tensors) / (len(self.Correct_Tensors) + len(self.InCorrect_Tensors))  
        self.current_accuracy = accuracy
        self.Correct_Tensors.clear()
        self.InCorrect_Tensors.clear()
        return accuracy
    
    def fit(self, inputs=[], outputs=[], epoch=1):
        if isinstance(self.Model._Data, Data):
            #self.dynamic_init()
            self.epoch = epoch
            if self.Model._Data.xtrain is not None:
                continue_training = True
                count_pass = 1
                while(continue_training):
                    #reset saved tensors              
                    for index in range(0, len(self.Model._Data.xtrain)) :
                        inp_ = self.Model._Data.xtrain[index]
                        op_ = self.Model._Data.ytrain[index]
                        self.pass_(inp_, op_)
                    accur = self.calculate_accuracy()
                    if accur < 0.9:
                        continue_training = True
                    # if continue_training == True:
                    #     self._update_weight()

                    print("Pass: "+str(count_pass) + " Accuracy: " + str(accur))
                    count_pass += 1
            