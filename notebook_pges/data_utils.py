import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py

from viz_utils import *
from numpy.lib.stride_tricks import as_strided

def read_data(path):
    datafile = h5py.File(path, 'r')
    timeseries = np.array(datafile['timeseries'])
    dates = np.array([d.decode("utf-8") for d in datafile['date']])
    return timeseries, dates

def add_bias_to_data(data):
    data += abs(np.nanmin(data))
    return data

def crop_volcanodata(timeseries, boundary, add_bias=True):
    ysize = boundary[0][1]-boundary[0][0]
    xsize = boundary[1][1]-boundary[1][0]
    nt = timeseries.shape[0]
    data = np.zeros((nt, ysize, xsize))
    for i, fr in enumerate(timeseries):
        data[i] = fr[boundary[0][0]:boundary[0][1], boundary[1][0]:boundary[1][1]]
    if (add_bias):
        data = add_bias_to_data(data)
    return data

def avrg_pool(data, kernel_size, stride, avrg_over_cols=False):
    # kernel_size : (ysize, xsize)
    # stride : (ysize, xsize)
    
    out_shape = ((data.shape[1] - kernel_size[0]) // stride[0] + 1,
                 (data.shape[2] - kernel_size[1]) // stride[1] + 1)
    
    data_strided = []
    for t in range(data.shape[0]):
        fr_data = data[t, :]
        stride_steps = (stride[0]*fr_data.strides[0], stride[1]*fr_data.strides[1]) + fr_data.strides
        data_strided.append(as_strided(fr_data, shape=out_shape + kernel_size, strides = stride_steps))
        
    data_strided = np.array(data_strided)
    assert (data_strided.shape[0] == data.shape[0])
    data_strided = np.nanmean(data_strided, axis=(3,4)).reshape(data.shape[0], out_shape[0], out_shape[1])
    
    # Replace lingering Nan's with average value of entire frame
    for t in range(data.shape[0]):
        data_strided[t] = np.nan_to_num(data_strided[t], nan=np.nanmean(data_strided[t]))
        
    if (avrg_over_cols):
        data_strided = np.nanmean(data_strided, axis=2)
    
    return data_strided
    
def extract_training_data(data, active_range, outer_range, grid_size=3):
    # active_range : ((y_range), (x_range))
    # outer_range : ((y_range), (x_range))
    assert ((outer_range[0][1] - outer_range[0][0]) % grid_size == 0)
    assert ((outer_range[1][1] - outer_range[1][0]) % grid_size == 0)
    
    # Remove activity piece
    for y in range(active_range[0][0], active_range[0][1]):
        for x in range(active_range[1][0], active_range[1][1]):
            data[:, y, x] = np.nan
       
    # Extract outer region around volcano
    data_outer = crop_volcanodata(data, outer_range)
    kernel_size = (int((outer_range[0][1] - outer_range[0][0]) / grid_size),
                   int((outer_range[1][1] - outer_range[1][0]) / grid_size)) # kernel_size : (y, x))

    return avrg_pool(data_outer, kernel_size, stride=kernel_size)
    
def reshape_input(array):
    # receives array (grid_size x grid_size)
    # returns array (1, grid_size x grid_size) = (n_samples, n_timesteps, n_features)
    return np.expand_dims(array.reshape(array.shape[0], array.shape[1]*array.shape[2]), axis=0)

def reshape_output(array, grid_size):
    # receives array (1, grid_size x grid_size)
    # returns array (grid_size x grid_size)
    array = np.squeeze(array)
    return array.reshape(grid_size, grid_size) 

def build_dataset(data, n_steps=9):
    # n_steps : number of steps in the input sequence
    # 3D format expected by LSTMs: [samples, timesteps, features]
    
    def series_to_supervised(sequences):
        dif, raw, dat = [], [], []
        
        for i, seq_d in enumerate(sequences['diff']): # seq shape: (1, n_time_steps, n_features)
            seq_r = sequences['raw'][i]
            dates = sequences['dates'][i]
            assert (seq_d.shape == seq_r.shape)
            assert (seq_d.shape[1] == len(dates))
            
            for t in range(0, seq_d.shape[1]-(n_steps)):
                dif.append(seq_d[:, t:t+(n_steps+1), :]) 
                raw.append(seq_r[:, t:t+(n_steps+1), :])
                dat.append(dates[t:t+(n_steps+1)])
        
        superv_dif = np.squeeze(np.array(dif), axis=1) # squeeze removes single-dimensional entries from the shape
        superv_raw = np.squeeze(np.array(raw), axis=1)
        dates = np.array(dat)
        
        assert (superv_dif.shape == superv_raw.shape)
        assert (dates.shape[0] == superv_dif.shape[0])
        
        return {'diff' : superv_dif, 'raw' : superv_raw, 'dates' : dates}
    
    def unison_shuffled_copies(a, b, c):
        # c is the 'dates' datastructure (2D)
        assert (a.shape == b.shape)
        assert (a.shape[0] == c.shape[0])
        n_rows = a.shape[0]
        p = np.random.permutation(n_rows)
        return a[p, :, :], b[p, :, :], c[p, :]
    
    def build_train_test_set(sequences):
        
        seq_sup = series_to_supervised(sequences) # dict of [n_samples, n_steps+1, n_features]
        
        # Shuffling sequences
        seq_sup['diff'], seq_sup['raw'], seq_sup['dates'] = unison_shuffled_copies(seq_sup['diff'], seq_sup['raw'], seq_sup['dates'])
        assert(seq_sup['diff'].shape == seq_sup['raw'].shape)
        assert(seq_sup['dates'].shape[0] == seq_sup['raw'].shape[0])
        
        # Building test set
        n_test = int(0.7 * seq_sup['raw'].shape[0])
        testset_raw = seq_sup['raw'][:n_test, :, :].copy() # Sample a few sequences from normal for test set
        testset_diff = seq_sup['diff'][:n_test, :, :].copy()
        testset_dates = seq_sup['dates'][:n_test].copy()
        testset_diff, testset_raw, testset_dates = unison_shuffled_copies(testset_diff, testset_raw, testset_dates)
        testset = {'diff' : testset_diff.copy(), 'raw' : testset_raw.copy(), 'dates' : testset_dates.copy()} 

        # Remove sequences added to test set from train
        trainset_diff = seq_sup['diff'].copy()
        trainset_diff = np.delete(trainset_diff, np.arange(n_test), axis=0)
        trainset_raw = seq_sup['raw'].copy()
        trainset_raw = np.delete(trainset_raw, np.arange(n_test), axis=0)
        trainset_dates = seq_sup['dates'].copy()
        trainset_dates = np.delete(trainset_dates, np.arange(n_test), axis=0)
        trainset = {'diff' : trainset_diff.copy(), 'raw' : trainset_diff.copy(), 'dates' : trainset_dates.copy()}
        
        assert (trainset['diff'].shape[0] == seq_sup['diff'].shape[0]-n_test)
        assert (trainset['diff'].shape == trainset['raw'].shape)
        assert (testset['diff'].shape == testset['raw'].shape)
       
        return {'train': trainset, 'test' : testset}
    
    # transform data to be stationary
    data['series_diff'] = np.diff(data['series'], axis=1)
    data['series'] = data['series'][:, 1:, :] # ignore first value of series ("eliminated" by diff)
    data['dates'] = data['dates'][1:]

    assert (len(data['dates']) == data['series_diff'].shape[1] == data['series'].shape[1])
    
    sequences = {'diff' : [data['series_diff']], 'raw' : [data['series']], 'dates' : [data['dates']]}

    return build_train_test_set(sequences)


def build_anomaly_map(data, lstm, grid_size=3, time_window=9): ## TODO: make this more efficient
    # Slicing data for inference and building anomaly map ("mse map")
    n_timest_data = data['diff'].shape[0]
    anomaly_map = np.zeros((data['raw'].shape[0]-time_window, data['raw'].shape[1], data['raw'].shape[2]))
    for t in range(time_window, n_timest_data-1): # scans until one timestep before last datapoint
        for y_ref in range(grid_size, data['diff'].shape[1]):
            for x_ref in range(grid_size, data['diff'].shape[2]):
                yrange = (y_ref-grid_size, y_ref)
                x_range = (x_ref-grid_size, x_ref)
                input_ar = data['diff'][t-time_window:t, yrange[0]:yrange[1], x_range[0]:x_range[1]]
                input_ar_raw = data['raw'][t-time_window:t, yrange[0]:yrange[1], x_range[0]:x_range[1]]
                gt_raw = np.expand_dims(data['raw'][t, yrange[0]:yrange[1], x_range[0]:x_range[1]], axis=0)

                # Reshaping
                input_ar = reshape_input(input_ar)
                input_ar_raw = reshape_input(input_ar_raw)
                gt_raw = reshape_input(gt_raw)

                # Inference 
                pred = lstm.predict(X=input_ar, 
                                    lastknown_obs=np.expand_dims(input_ar_raw[:, -1,:], axis=1))
                difference =  abs(pred-gt_raw)
                anomaly_map[t-time_window, yrange[0]:yrange[1], x_range[0]:x_range[1]] = reshape_output(difference, grid_size)
    
    return anomaly_map

