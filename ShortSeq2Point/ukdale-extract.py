from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from shortseq2pointdisaggregator import ShortSeq2PointDisaggregator
import metrics


def _normalize(chunk, mmax):
    '''Normalizes timeseries

    Parameters
    ----------
    chunk : the timeseries to normalize
    max : max value of the powerseries

    Returns: Normalized timeseries
    '''
    tchunk = chunk / mmax
    return tchunk


print("========== OPEN DATASETS ============")
train = DataSet(
    '/Users/mj/Documents/Learning_Stuff/NILM/neural-disaggregator/ShortSeq2Point/ukdale.h5')
test = DataSet(
    '/Users/mj/Documents/Learning_Stuff/NILM/neural-disaggregator/ShortSeq2Point/ukdale.h5')

train.set_window(start="13-4-2013", end="1-1-2014")
test.set_window(start="1-1-2014", end="30-3-2014")

window_size = 100
train_building = 1
test_building = 1
sample_period = 6
meter_key = 'kettle'
train_elec = train.buildings[train_building].elec
test_elec = test.buildings[test_building].elec

train_meter = train_elec.submeters()[meter_key]
train_mains = train_elec.mains()
test_mains = test_elec.mains()
# Here we get a generator back...
train_main_power_series = train_mains.power_series(sample_period=sample_period)
train_meter_power_series = train_meter.power_series(
    sample_period=sample_period)
test_mains_power_series = test_mains.power_series(sample_period=sample_period)
# There is only one chunk of data...so next() retrieves it all.
# We are returned Pandas dataframes.
df_main = next(train_main_power_series)
df_meter = next(train_meter_power_series)
df_test = next(test_mains_power_series)
#

df_main.to_pickle(
    'data/UK_Dale_aggregate_train_Omar.pkl.zip', compression='zip')
df_meter.to_pickle('data/UK_Dale_kettle_Omar.pkl.zip', compression='zip')
df_test.to_pickle(
    'data/UK_Dale_aggregate_test_Omar.pkl.zip', compression='zip')
