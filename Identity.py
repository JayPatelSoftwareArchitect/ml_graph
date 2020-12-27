# Properties like ids will be used to identify a single instance
# this will be inherited where applicable
import typing
# import uuid


class Identity(object):
    '''An id class for consistancy in all classes with unique identity'''

    def __init__(self):
        self.__Id = id(self)  # address of itself

    def get_Id(self) -> int:
        return self.__Id

    def set_Id(self, Id: int):
        self.__Id = Id
