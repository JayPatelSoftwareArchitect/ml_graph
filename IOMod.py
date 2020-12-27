import SharedCounter


class TInput(object):

    def __init__(self, input_, shape=None):
        self.dist = {}
        self.shape = shape

        attr = SharedCounter.Counter
        if isinstance(input_, str):
            for index in range(len(input_)):
                self.dist[index + 1] = int(input_[index])

        elif isinstance(input_, (dict, list, set)):
            for inp in input_:
                self.dist[attr] = inp
                attr += 1
        elif isinstance(input_, (float, int)):
            self.dist[attr] = input_

        else:
            raise Exception(
                "Check input type. It has to be one of (int, float, string, list, set, dict)")


class TOutput(object):
    def __init__(self, output_, shape=None, Lables=None):
        self.dist = {}
        self.shape = shape
        self.Lables = []
        if Lables not None:
            for each in Lables:
                self.Lables.append(each)

        attr = SharedCounter.Counter
        if isinstance(output_, str):
            for index in range(len(output_)):
                self.dist[attr] = int(output_[index])
                attr += 1
        elif isinstance(output_, dict, set, list):
            for opt in output_:
                self.dist[attr] = opt

        elif isinstance(output_, (float, int)):
            self.dist[attr] = output_

        else:
            raise Exception(
                "Check output type. It has to be one of (int, float, string, list, set, dict)")


class LoadModel():
    def __init__(self, file_path="model.txt"):
        Weights = []
        Bais = []
        Num_Of_Containers = 0
        Num_Of_TensorsInEach = []
        file_model = open(file_path, 'r')
        all_lines = file_model.readlines()
        for line_index in range(len(all_lines)):
            line = all_lines[line_index]
            if line == "Container_Length_\n":
                line_index += 1
                while all_lines[line_index] == "":
                    line_index += 1
                Num_Of_Containers = int(all_lines[line_index])

            elif line == "LastTensorId_\n":
                line_index += 1
                while all_lines[line_index] == "":
                    line_index += 1
                Num_Of_TensorsInEach.append(int(all_lines[line_index]))

            elif line == "Weight_\n":
                line_index += 1
                while all_lines[line_index] == "":
                    line_index += 1
                Weights.append(float(all_lines[line_index]))

            elif line == "Bais_\n":
                line_index += 1
                while all_lines[line_index] == "":
                    line_index += 1
                Bais.append(float(all_lines[line_index]))

        file_model.close()

        model = MultiContainer.MultiContainer(
            Num_Of_Containers, Num_Of_TensorsInEach, [0])
        model.InitWeights(Weights, Bais)

        self.Model = model


class SaveModel():
    def __init__(self, Model, file_path="model1.txt"):
        if isinstance(Model, MultiContainer.MultiContainer):
            file_model = open(file_path, 'w')
            lines = []
            lines.append("MULTI_CONTAINER_ID\n" +
                         str(Model.MULTI_CONTAINER_ID)+"\n")
            lines.append("Container_Length_\n" +
                         str(Model.Container_Length_)+"\n")
            for x in Model.Container_dist:
                layer = Model.Container_dist[x]
                lines.append("Container\n" + str(layer.Id_)+"\n" +
                             "FristTendorId_\n"+str(layer.FristTendorId_)+"\n"
                             + "LastTensorId_\n"+str(layer.LastTensorId_)+"\n")
                for tensor_id in layer.Tensors_dist:
                    tensor = layer.Tensors_dist[tensor_id]
                    lines.append("Tensor\n"+str(tensor.Id_)+"\n" +
                                 "Weight_\n"+str(tensor.Weight_)+"\n" +
                                 "Bais_\n"+str(tensor.Bais_)+"\n"
                                 )
            file_model.writelines(lines)
            file_model.close()
