from multiprocessing import Pool
import time
import os

class Execute():
    def __init__(self):
        _ = 0

    def worker(self,func, args):
        func(*args)


    def multiProcessTasks(self, NUMBER_OF_PROCESSES, tasks = [] ):
        #task would be an array of tuple of function and args
        with Pool(NUMBER_OF_PROCESSES) as pool:
            for task in tasks:
                pool.apply_async(self.worker, task)
            