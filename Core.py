from GraphImport import *
from Training import Training
from Optimizer import Callback, Random
import numpy as np

model = Model(NumLayer=6, NumTensor=[784, 20, 30, 15, 10, 10])
