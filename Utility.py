import typing
from Identity import Identity


class Utility(object):
    @staticmethod
    def T_Connect_Next(Instance_A: object, Instance_B: object):
        if hasattr(Instance_A, 'N_Connected') and isinstance(Instance_A, Identity):
            if isinstance(Instance_B, (set, dict, list)):
                for _instance in Instance_B:
                    if isinstance(_instance, Identity):
                        Instance_A.N_Connected[_instance.get_Id()] = _instance

            else:
                Instance_A.N_Connected[Instance_B.get_Id()] = Instance_B

    @staticmethod
    def T_Connect_Previous(Instance_A: object, Instance_B: object):
        if hasattr(Instance_A, 'P_Connected') and isinstance(Instance_A, Identity):
            if isinstance(Instance_B, (set, dict, list)):
                for _instance in Instance_B:
                    if isinstance(_instance, Identity):
                        Instance_A.P_Connected[_instance.get_Id()] = _instance

            else:
                Instance_A.P_Connected[Instance_B.get_Id()] = Instance_B


# class Minst():
#     def input_single_daimentional_array(input):
#         single_diamentional_input = []
#         for x in range(len(input)):
#             for y in range(len(input[x])):
#                 single_diamentional_input.append(input[x][y])

#         return single_diamentional_input
