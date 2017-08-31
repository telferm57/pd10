# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:56:26 2017

@author: dad
"""
import cfg
from  mhealthx.xio import read_accel_json
from  mhealthx.signals import autocorrelate
imp.reload(mhealthx.signals)
from mhealthx.extract import run_pyGait
from mhealthx.extractors.pyGait import project_walk_direction_preheel, walk_direction_preheel
import pandas as pd
import importlib as imp
import mhealthx.extractors.pyGait
imp.reload(mhealthx.extractors.pyGait)
from mhealthx.extractors.pyGait import gait
imp.reload(mhealthx.extractors.pyGait)
from mhealthx.extractors.pyGait import heel_strikes, gait
from mhealthx.extract import make_row_table

input_file = cfg.WRITE_PATH + '/testfiles/' + 'accel_walking_return-test-1.csv'
input_file = "C:\\Users\\dad\\.synapseCache\\993\\5402993\\993\\5402993\\accel_walking_outbound.json.items-c76b8dee-24d9-4252-84b3-71f5e9d1becc7163809297980866330.tmp"
basePath = "C:\\Users\\dad\\.synapseCache\\428\\5389428\\428\\5389428\\"

tf2 = "accel_walking_outbound.json.items-502d0501-43ee-4462-a7e4-012e216576922296422479994601681.tmp"
input_file = basePath + tf2
start = 150
device_motion = False
t, axyz, gxyz, uxyz, rxyz, sample_rate, duration = read_accel_json(input_file, start, device_motion)
ax, ay, az = axyz
len(axyz)
type(axyz)

sum(az)
stride_fraction = 1.0/8.0
threshold0 = 0.5
threshold = 0.2
order = 4
cutoff = max([1, sample_rate/10])
distance = None
row = pd.Series({'a':[1], 'b':[2], 'c':[3]})
file_path = 'test1.csv'
table_stem = cfg.WRITE_PATH + '/testfiles/features'
save_rows = True
#walk_direction_preheel
#def walk_direction_preheel(ax, ay, az, t, sample_rate, 
#                          stride_fraction=1.0/8.0, threshold=0.5,
#                         order=4, cutoff=5, plot_test=False)
import os
os.path.basename(file_path)
try:
    strikes, strike_indices = heel_strikes(pz, sample_rate, threshold,
                                           order, cutoff, False, t)
except ValueError as e:
    print("got an error: %s"%e)
else:
    print("went fne")
 
number_of_steps, cadence, velocity, avg_step_length, avg_stride_length,\
step_durations, avg_step_duration, sd_step_durations, strides, \
stride_durations, avg_number_of_strides, avg_stride_duration, \
sd_stride_durations, step_regularity, stride_regularity, \
symmetry = gait(strikes, data, duration, distance)

directions = walk_direction_preheel(ax, ay, az, t, sample_rate,\
                                            stride_fraction, threshold0, order, cutoff)
import numpy as np
np.tile(directions,(3,1))

px, py, pz = project_walk_direction_preheel(ax, ay, az, t, sample_rate,\
                                            stride_fraction, threshold0, order, cutoff)
len(px)

feature_row, feature_table = run_pyGait(py, t, sample_rate,
                                        duration, threshold, order,
                                        cutoff, distance, row, file_path,
                                        table_stem, save_rows)