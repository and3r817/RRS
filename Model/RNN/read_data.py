from __future__ import print_function

import numpy as np
import pandas as pd
import helpers.command_parser as cp
import helpers.command_parser as parse
import helpers.early_stopping as EsParse
from helpers.data_handling import DataHandler

dataset = DataHandler(dirname="ks-cooks-1y")

print(dataset.dirname)


