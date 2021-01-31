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
    #any coptimizer which call function that takes in 3 inputs, max tensors that were output , min tensors that were outputed by model, 
    def __init__(self, model,SavedTensors=None ,optimizer='',activationfn=None, threshold=None):
        if optimizer == 'random':
            self.Optimizer = Random(SavedTensors=SavedTensors,activationfn=activationfn)
        elif optimizer == 'gradientdecent':
            self.Optimizer = BackProp(model=model,SavedTensors=SavedTensors)

class BackProp(object):
    def __init__(self, model ,SavedTensors):
        self.accuracy = 0.0
        self.seen_set = set()
        self.SavedTensors = None
        self.Model = model
        self.alpha = 0.1
        self.CorrectOnes = dict()
        if isinstance(SavedTensors, SaveTensors):
            self.SavedTensors = SavedTensors
            self.SavedTensors.set_last_nodes()
    
    def addCorrect(self,pass_,tnode):
        self.CorrectOnes[pass_] = tnode

    def weight_change(self, loss, activation , weight):
        return weight - (self.alpha * (activation * loss ))

    def bais_change(self, loss, activation , bais):
        return bais + (self.alpha * (loss * activation))


    def call(self):
        #j` = z.delta  z = current activation delta = loss of the unit 
        self.seen_set.clear()
        # each layer of model has a pass counter which will be the key of loss and activation for each tnode and each wt instance
        # setup is in such a way that those values will be updated as model passes through inputs.
        # So here just the calculation of weight change will be left. 
        start_c = 1
        pass_continue = True
        if isinstance(self.Model, Model):
            while pass_continue == True:
                correct_one = self.CorrectOnes[start_c]
                l_id = None
                for layer_id in self.Model.Container:
                    l_id = layer_id
                layer = self.Model.Container[l_id]
                while True:    
                    for t_node_id in layer.Container:
                        t_node = layer.Container[t_node_id]
                        if t_node != correct_one:
                            for each_wt_id in t_node.P_ConnectedWt:
                                wt = t_node.P_ConnectedWt[each_wt_id]
                                loss_, activation_ = wt._weight_change_call(start_c)
                                new_weight = self.weight_change(loss_, activation_, wt.get_NodeWeight())
                                wt.set_NodeWeight(new_weight)
                                new_bais = self.bais_change(loss_,activation_, wt.get_NodeBais())
                                wt.set_NodeBais(new_bais)
                                #print("Wt_Id: " + str(each_wt_id) + "weight: "+ str(new_weight) + "bais: "+ str(new_bais))
                        else:
                            for each_wt_id in t_node.P_ConnectedWt:
                            
                                wt = t_node.P_ConnectedWt[each_wt_id]
                            
                                new_weight = wt.get_NodeWeight() + self.alpha*(wt.get_ActivationVal() + wt.get_NodeBais()) 
                                wt.set_NodeWeight(new_weight)

                    if len(layer.P_Connected) == 0:
                        break
                    for previousL in layer.P_Connected:
                        layer = layer.P_Connected[previousL]

                #check the last pass number to be same as current and if yes , close this loop
                if self.Model.Container[l_id].pass_counter - 1 != start_c:
                    start_c += 1
                else:
                    self.Model.Container[l_id].pass_counter = 1
                    pass_continue = False
            

class Random(object):

    def __init__(self,SavedTensors, activationfn=None, threshold=None):
        self.accuracy = 0.0
        self.previous_accuracy = 0.0
        self.depth = 1
        self.seen_set = set()
        self.threshold = activationfn.threshold
        self.activationfn = activationfn
        self.change_counter = 2
        self.all_tnodes = []
        self.SavedTensors = None
        if isinstance(SavedTensors, SaveTensors):
            self.SavedTensors = SavedTensors

    def call(self):
        self.seen_set.clear()
        self.depth = 1
        allmax_connected = dict()
        allmin_connected = dict()
        max_tensors=self.SavedTensors.get_all_max_tensors()
        min_tensors=self.SavedTensors.get_all_min_tensors()
        # for k in max_tensors:
        #     for j in max_tensors[k].P_Connected:
        #         allmax_connected[max_tensors[k].P_Connected[j].get_Id()] = max_tensors[k].P_Connected[j]
        # for k in min_tensors:
        #     for j in min_tensors[k].P_Connected:
        #         allmin_connected[min_tensors[k].P_Connected[j].get_Id()] = min_tensors[k].P_Connected[j]
        seen_set=self.seen_set
        self.optimize_randomly(max_tensors=max_tensors, min_tensors=min_tensors, seen_set=seen_set)
    
    def apply_change(self, wt,op='+'):
        random_index = random.randrange(0, len(wt))
        if op == '+':
            wt[random_index] += SharedCounter.WT_CHANGE
        elif op == '-':
            wt[random_index] -= SharedCounter.WT_CHANGE

    def random_change(self, weight_instance, op='+'):
        if isinstance(weight_instance, Weight):
            in_ = weight_instance.get_NodeInput() 
            wt = weight_instance.get_NodeWeight()
            bais_ = weight_instance.get_NodeBais()
            current_activation = self.activationfn._calculate(bais_, wt, in_)
            if op == '+':
                #while current_activation < self.threshold-1:
                self.apply_change(wt, op=op)
                self.apply_change(bais_, op=op)
                current_activation = self.activationfn._calculate(bais_, wt, in_)
            elif op == '-':
                #while current_activation > self.threshold + 1:
                self.apply_change(wt, op=op)
                self.apply_change(bais_, op=op)
                    
                current_activation = self.activationfn._calculate(bais_, wt, in_)
      
            # self.depth += 1    

    def optimize_randomly(self, max_tensors=None, min_tensors=None,seen_set=None):
        
        # will try to reduce weights of max tensors , and increse weights of min_tensor randomly
        #if a dict
        if isinstance(max_tensors, dict):
            for max_ in max_tensors:
                max_tensor = max_tensors[max_]
                if isinstance(max_tensor, TNode):
                    #if last_node == False:
                    if max_tensor.P_ConnectedWt is not None:
                        #iterate through all previous weights .
                        for wt_ in max_tensor.P_ConnectedWt:
            
                            weight_instance = max_tensor.P_ConnectedWt[wt_]
                            #weight_instance.set_Bais(weight_instance.get_Bais() - SharedCounter.RANDOM_STEP)
                        #    if last_node == False:
            
                            self.random_change(weight_instance, '-')
            
                            if seen_set.__contains__(weight_instance) is False:
                                seen_set.add(weight_instance)
                                #print(weight_instance.TNode.get_Id())
                                self.optimize_randomly(max_tensors=weight_instance.TNode, seen_set=seen_set)
        elif isinstance(max_tensors, TNode):
            max_tensor = max_tensors
            #max_tensor.set_Bais(max_tensor.get_Bais() - SharedCounter.RANDOM_STEP)
            if max_tensor.P_ConnectedWt is not None:
            #iterate through all previous weights .
                for wt_ in max_tensor.P_ConnectedWt:
            
                    weight_instance = max_tensor.P_ConnectedWt[wt_]
                        #    if last_node == False:
            
                    self.random_change(weight_instance, '-')
            
                    if seen_set.__contains__(weight_instance) is False:
                        seen_set.add(weight_instance)
                                #print(weight_instance.TNode.get_Id())
                        self.optimize_randomly(max_tensors=weight_instance.TNode, seen_set=seen_set)
        
        if isinstance(min_tensors, dict):
            for min_ in min_tensors:
                min_tensor = min_tensors[min_]
                if isinstance(min_tensor, TNode):
                    #if last_node == False:
                    #min_tensor.set_Bais(min_tensor.get_Bais() - SharedCounter.RANDOM_STEP)
                    if min_tensor.P_ConnectedWt is not None:
                        #iterate through all previous weights .
                        for wt_ in min_tensor.P_ConnectedWt:
                            weight_instance = min_tensor.P_ConnectedWt[wt_]
                            #if last_node == False:
                            self.random_change(weight_instance, '+')
            
                        
                            if seen_set.__contains__(weight_instance) is False:
                                seen_set.add(weight_instance)
                                #print(weight_instance.TNode.get_Id())
                                self.optimize_randomly(min_tensors=weight_instance.TNode, seen_set=seen_set)
        elif isinstance(min_tensors, TNode):
                #if last_node == False:
            min_tensor = min_tensors
            #min_tensor.set_Bais(min_tensor.get_Bais() - SharedCounter.RANDOM_STEP)
            if min_tensor.P_ConnectedWt is not None:
                #iterate through all previous weights .
                for wt_ in min_tensor.P_ConnectedWt:
                    weight_instance = min_tensor.P_ConnectedWt[wt_]
                        #if last_node == False:
                    self.random_change(weight_instance, '+')
    
                    
                    if seen_set.__contains__(weight_instance) is False:
                        seen_set.add(weight_instance)
                            #print(weight_instance.TNode.get_Id())
                        self.optimize_randomly(min_tensors=weight_instance.TNode, seen_set=seen_set)
    
        