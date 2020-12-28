from MutiProcess import Execute
from Identity import Identity


class Container(Execute):
    def __init__(self):
        self.Container = dict()
        Execute.__init__(self)

    def add(self, Instance):
        if isinstance(Instance, Identity):
            self.Container[Instance.get_Id()] = Instance

    def remove(self, Instance_Id):
        if Instance_Id in self.Container:
            del self.Container[Instance_Id]

    def callOnEach(self,NUMBER_OF_PROCESSES, TASKS=[]):
        self.multiProcessTasks(NUMBER_OF_PROCESSES, TASKS)