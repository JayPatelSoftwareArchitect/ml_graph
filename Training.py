from TModel import Model
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


class Training(object):
    def __init__(self, model):
        if isinstance(model, Model):
            self.Model = model
            self.Output_Options = []
            self.Input_Options = []
            self.Callback = None
        else :
            raise RuntimeError("passed model is not of type Model")

    def sample(Input_Options=[], Output_Options=[]):
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
        #number_of_selection is an int of howmany tensors will be adjusted per each callIter() default=0 will add all tensors of model
        if default == 0:
            self.Callback = Callback()
            #add all tensors
            for layer_id in self.Model.Container:
                layer = self.Model.Container[layer_id]
                for tensor_id in layer.Container:    
                    tensor = layer.Container[tensor_id]
                    self.Callback.addNode(tensor)
            self.Callback.set_EachOp(number_of_selection)



    def fit(inputs=[], outputs=[]):
        for index in range(0, len(inputs)) :
            inp_ = inputs[index]
            op_ = outputs[index]