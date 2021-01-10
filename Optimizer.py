from GraphImport import *
#a wraped call back class which will have access to all tensor nodes, and will have an optimizer        

class Callback(object):
    def __init__(self):
        self.TNode = []
        self.Flags = []
        self.TNodeCounter = 0
        self.TNodeSelect = 0
        self.Optimizer = None
    def addNode(self, node, flag=False):
        self.TNode.append(node)
        self.Flags.append(flag)

    def remove(self):
        self.TNode.pop()
    
    def set_EachOp(self, select):
        self.TNodeSelect = select


    def callIter(self):
        for _ in range(0,self.TNodeSelect):
            tensor = self.TNode[self.TNodeCounter]
            flag = self.Flags[self.TNodeCounter]
            #tensor._CallFn()
            self.TNodeCounter+=1
            if self.TNodeCounter > len(self.TNode):
                self.TNodeCounter = 0
        # simple algo which will select only one tensor of the model on each training example
        # It will select weights such that it can activate that tensor if current activation value 
        # is less then threshold value.

class Optimizer(object):
    def __init__(self, optimizer=''):
        if optimizer == 'random':
            self.Optimizer = Random()

class Random(object):

    def __init__(self):
        self.accuracy = 0.0
        self.previous_accuracy = 0.0
        self.seen_set = set()
    def call(self, max_tensor=None, min_tensor=None, last_node=False):
        self.seen_set.clear()
        self.optimize_randomly(max_tensors=max_tensor,min_tensors=min_tensor, seen_set=self.seen_set,last_node=last_node)
  
    def random_change(self, weight_instance, op='+'):
        if isinstance(weight_instance, Weight):
            wt = weight_instance.get_NodeWeight()
            if isinstance(wt, list):
                new_weight = []
                if op == '+':
                    for i in range(0, len(wt)):
                        new_weight.append(wt[i] + random.uniform(SharedCounter.RANDOM_STEP, SharedCounter.RANDOM_STEP_END))
                elif op == '-':
                    for i in range(0, len(wt)):
                        new_weight.append(wt[i] -random.uniform(SharedCounter.RANDOM_STEP, SharedCounter.RANDOM_STEP_END))

                return new_weight
            else:
                if op == '+':
                    return weight_instance.get_NodeWeight() + random.uniform(SharedCounter.RANDOM_STEP, SharedCounter.RANDOM_STEP_END)
                elif op == '-':
                    return weight_instance.get_NodeWeight() - random.uniform(SharedCounter.RANDOM_STEP, SharedCounter.RANDOM_STEP_END)

    def optimize_randomly(self, max_tensors=None, min_tensors=None, seen_set=None, last_node=False):
        # will try to reduce weights of max tensors , and increse weights of min_tensor randomly
        #if a dict
        if isinstance(max_tensors, dict):
            for max_ in max_tensors:
                max_tensor = max_tensors[max_]
                if isinstance(max_tensor, TNode):
                    #if last_node == False:
                    max_tensor.set_Bais(max_tensor.get_Bais() - random.uniform(SharedCounter.RANDOM_STEP, SharedCounter.RANDOM_STEP_END))
                    if max_tensor.P_ConnectedWt is not None:
                        #iterate through all previous weights .
                        for wt_ in max_tensor.P_ConnectedWt:
            
                            weight_instance = max_tensor.P_ConnectedWt[wt_]
                        #    if last_node == False:
            
                            weight_instance.set_NodeWeight(self.random_change(weight_instance, '-'))
            
                            if seen_set.__contains__(weight_instance) is False:
                                seen_set.add(weight_instance)
                                #print(weight_instance.TNode.get_Id())
                                self.optimize_randomly(weight_instance.TNode, seen_set=seen_set)
        elif isinstance(max_tensors, TNode):
            max_tensor = max_tensors
            max_tensor.set_Bais(max_tensor.get_Bais() - random.uniform(SharedCounter.RANDOM_STEP, SharedCounter.RANDOM_STEP_END))
            if max_tensor.P_ConnectedWt is not None:
            #iterate through all previous weights .
                for wt_ in max_tensor.P_ConnectedWt:
            
                    weight_instance = max_tensor.P_ConnectedWt[wt_]
                        #    if last_node == False:
            
                    weight_instance.set_NodeWeight(self.random_change(weight_instance, '-'))
            
                    if seen_set.__contains__(weight_instance) is False:
                        seen_set.add(weight_instance)
                                #print(weight_instance.TNode.get_Id())
                        self.optimize_randomly(weight_instance.TNode, seen_set=seen_set)
        
        if isinstance(min_tensors, dict):

            for min_ in min_tensors:
                min_tensor = min_tensors[min_]
                if isinstance(min_tensor, TNode):
                    #if last_node == False:
                    min_tensor.set_Bais(min_tensor.get_Bais() - random.uniform(SharedCounter.RANDOM_STEP, SharedCounter.RANDOM_STEP_END))
                    if min_tensor.P_ConnectedWt is not None:
                        #iterate through all previous weights .
                        for wt_ in min_tensor.P_ConnectedWt:
                            weight_instance = min_tensor.P_ConnectedWt[wt_]
                            #if last_node == False:
                            weight_instance.set_NodeWeight(self.random_change(weight_instance, '+'))
            
                        
                            if seen_set.__contains__(weight_instance) is False:
                                seen_set.add(weight_instance)
                                #print(weight_instance.TNode.get_Id())
                                self.optimize_randomly(weight_instance.TNode, seen_set=seen_set)
        elif isinstance(min_tensors, TNode):
                #if last_node == False:
            min_tensor = min_tensors
            min_tensor.set_Bais(min_tensor.get_Bais() - random.uniform(SharedCounter.RANDOM_STEP, SharedCounter.RANDOM_STEP_END))
            if min_tensor.P_ConnectedWt is not None:
                #iterate through all previous weights .
                for wt_ in min_tensor.P_ConnectedWt:
                    weight_instance = min_tensor.P_ConnectedWt[wt_]
                        #if last_node == False:
                    weight_instance.set_NodeWeight(self.random_change(weight_instance, '+'))
        
                    
                    if seen_set.__contains__(weight_instance) is False:
                        seen_set.add(weight_instance)
                            #print(weight_instance.TNode.get_Id())
                        self.optimize_randomly(weight_instance.TNode, seen_set=seen_set)
    
        