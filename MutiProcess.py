from multiprocessing import Pool
import time
import os
import concurrent.futures

class Execute():
    def __init__(self):
        _ = 0

    def worker(self,func, args):
        func(*args)


    def multiProcessTasks(self, NUMBER_OF_PROCESSES, tasks = [] ):
        #task would be an array of tuple of function and args
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ = set()
            for task in range(0, len(tasks)):
                tuple_ = tasks[task]
                fnc = tuple_[0]
                if len(tuple_) > 1:
                    arg_ = tuple_[1]
                    a = executor.submit(fnc,arg_ )
                    future_to_.add(a)
                else :
                    a = executor.submit(fnc)
                    future_to_.add(a)
            for future in concurrent.futures.as_completed(future_to_):
                try:
                    data = future.result()
                except Exception as exc:
                    print("generated an exception")
                

