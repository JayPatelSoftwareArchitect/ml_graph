import typing
import SharedCounter


class Position(object):
    '''Defines Position in Model or Position in Layer'''

    def __init__(self):
        self.Position = SharedCounter.UNSET

    def get_Position(self) -> int:
        return self.Position

    def set_PositionAsStart(self):
        self.Position = SharedCounter.START

    def set_PositionAsEnd(self):
        self.Position = SharedCounter.END

    def set_PositionAsHidden(self):
        self.Position = SharedCounter.HIDDEN

    def islastLayerNode(self):
        if self.Position == SharedCounter.END:
            return True
        else:
            return False
    def isFirstLayer(self):
        if self.Position == SharedCounter.START:
            return True
        else:
            return False