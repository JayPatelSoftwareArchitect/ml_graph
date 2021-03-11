import sys

sys.path.append(sys.path[0]+'/graph')

from TModel import Model
import tensorflow as tf
import hashlib
from Utility import Utility
from functools import cmp_to_key
from Data import Data
from TNode import TNode
import SharedCounter
import random
from Internals import Weight
from IOMod import SaveTensors

