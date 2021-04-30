import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from viz_utils import *

class lstm:
    def __init__(self, n_repeat=1, 
                 n_in=6, n_out=6, 
                 n_feat=1, n_batch=1, 
                 n_epoch=200):
        self.n_repeat = n_repeat
        self.n_in = n_in
        self.n_out = n_out
        self.n_feat = n_feat
        self.n_epoch = n_epoch
        self.n_batch = n_batch
    
    def preprocess_traindata(self, train_data):
        
        def series_to_supervised(series):
            df = pd.DataFrame(series)
            cols = list()

            # input sequence (t-n, ..., t-1)
            for i in range(self.n_in, 0, -1):
                cols.append(df.shift(i))
            # forecast sequence (t, t+1, ..., t+n)
            for i in range(0, self.n_out):
                cols.append(df.shift(-i))

            agg = pd.concat(cols, axis=1)
            agg.dropna(inplace=True)

            return agg.values
        
        # transform data to be stationary
        diff_series = np.diff(train_data)
        # transform series to supervised problem
        series_sup = series_to_supervised(diff_series)
        series_sup_raw = series_to_supervised(train_data[1:]) # ignore first value of series ("eliminated" by diff)
        # shuffle dataset
        series_sup, series_sup_raw = self.unison_shuffled_copies(series_sup, series_sup_raw)
        
        return series_sup, series_sup_raw
    
    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def scale_train(self, df_train):
        nsamples, ntimesteps, nfeatures = df_train.shape
        
        # reshape to use sklearn scaler
        df_train_rs = df_train.copy().reshape((nsamples, ntimesteps * nfeatures))
        
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = scaler.fit(df_train_rs)
        
        # reshape back
        return scaler.transform(df_train_rs).reshape((nsamples, ntimesteps, nfeatures))
    
    def scale_test(self, df_test):
        return self.scaler.transform(df_test)
    
    def invert_scale(self, X, yhat):
        new_row = np.concatenate((X, yhat), axis=1)
        nsamples, ntimesteps, nfeatures = new_row.shape
        
        # reshape to match sklearn input
        new_row = new_row.reshape((nsamples, ntimesteps * nfeatures))
        inverted = self.scaler.inverse_transform(new_row).reshape((nsamples, ntimesteps, nfeatures))
        
        return inverted[:, -(self.n_out):, :]
        
    def train_lstm(self, train_data, train_data_raw):
        train = self.scale_train(train_data)
        train_raw = train_data_raw.copy()

        assert (train.shape == train_raw.shape)

        X, y = train[:, :(self.n_in), :], train[:, (self.n_in):, :]

        hist = []
        for j in range(self.n_repeat):
            train_rmse = list()
            self.build_model()
            for i in range(self.n_epoch):
                self.model.fit(X, y, 
                               epochs=1, 
                               batch_size=1, 
                               verbose=0, 
                               shuffle=True)
                self.model.reset_states()
                train_rmse.append(self.evaluate(X=train[:, :(self.n_in), :], 
                                                lastknown_obs=np.expand_dims(train_raw[:, (self.n_in-1), :], axis=1), 
                                                gt=train_raw[:, (self.n_in):, :]))
            hist.append({'train' : train_rmse})
        return hist
        
    def predict(self, X, lastknown_obs):
        output = self.model.predict(X, batch_size=1)
        output = np.expand_dims(output, axis=1) # Match shape of input - output originally in 2D since return_sequences=False
        
        if type(lastknown_obs) is np.ndarray:
            assert(output.shape[0] == lastknown_obs.shape[0])
        
        predictions = list()
        for i in range(output.shape[0]): # go over every row of predicted output
            yhat = np.expand_dims(output[i, :, :], axis=0) # make the slice of output 3d
            # invert scaling
            X_i = np.expand_dims(X[i], axis=0)  # make row of input 3D to match rnn input
            yhat = self.invert_scale(X_i, yhat)
            
            # invert differencing
            lastknown = np.expand_dims(lastknown_obs[i, :, :], axis=0) # make it 3d to match yhat
            yhat = np.concatenate((lastknown, yhat), axis=1) # concatenate last number of input sequence with prediction to invert differencing
            yhat = yhat.cumsum(axis=1) # invert differencing using cumulative sum over time step axis 

            # the last observed value is only used to invert the differencing operation
            predictions.append(yhat[:, 1:, :])
        
        predictions = np.array(predictions).reshape(len(predictions), -1, self.n_feat)
        return predictions

    def train_and_validate_lstm(self, train, train_raw, 
                                valid, valid_raw):
        
        train = self.scale_train(train)
        train_raw = train_raw.copy()
        
        X, y = train[:, :(self.n_in), :], train[:, (self.n_in):, :]
    
        # Training and evaluating
        hist = []
        self.model_save = None # Saves the model with smallest rmse on validation set
        self.min_rmse = np.inf
        for j in range(self.n_repeat):
            train_rmse, test_rmse = list(), list()
            self.build_model()
            for i in range(self.n_epoch):
                #if (i % 10 == 0):
                #    print (f"Training iteration {j+1}, epoch {i}")
                self.model.fit(X, y, 
                               epochs=1, 
                               batch_size=1, 
                               verbose=0, 
                               shuffle=True)
                self.model.reset_states()
                # evaluate model on train data
                train_rmse.append(self.evaluate(X=train[:, :self.n_in, :], 
                                                lastknown_obs=np.expand_dims(train_raw[:, (self.n_in-1), :], axis=1), 
                                                gt=train_raw[:, (self.n_in):, :]))
                self.model.reset_states()
                # evaluate model on validation data
                test_rmse.append(self.evaluate(X=valid[:, :self.n_in, :], 
                                               lastknown_obs=np.expand_dims(valid_raw[:, (self.n_in-1), :], axis=1), 
                                               gt=valid_raw[:, (self.n_in):, :]))
                if (test_rmse[-1] < self.min_rmse):
                    self.min_rmse = test_rmse[-1]
                    self.model_save = self.model
                    
                self.model.reset_states()

            hist.append({'train' : train_rmse, 'valid' : test_rmse})
            
        #self.plot_results(hist)

        ## Assign model with lowest mse on test set
        self.model = self.model_save
    
    def evaluate(self, X, lastknown_obs, gt):
        predictions = self.predict(X, lastknown_obs)
    
        # make arrays 2d for sklearn's mean_squared_error
        return sqrt(mean_squared_error(np.squeeze(gt), np.squeeze(predictions)))
        
    def build_model(self):
        self.model = Sequential()
        # units : positive integer, dimensionality of the output space. https://keras.io/api/layers/recurrent_layers/lstm/
        # return sequences: This argument tells whether to return the output at each time steps instead of the final time step.
        self.model.add(LSTM(units=self.n_feat, 
                            batch_input_shape=(self.n_batch, self.n_in, self.n_feat), 
                            stateful=True, 
                            return_sequences=False))
        self.model.add(Dense(self.n_feat))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        # self.model.summary()
    
    def scale_test(self, df_test):
        return self.scaler.transform(df_test)
    
    def plot_test_results (self, eval_seq, title):
        '''
            eval_seq: dict of tuples - each entry is a sequence to be evaluated. 
            The first entry on tuple is the 'diff' version of the sequence, and 
            the second entry is the 'raw' version of the sequence 
        '''
        fig = plt.figure()
        for key in eval_seq:
            rmse_val = []
            for row in range(eval_seq[key][0].shape[0]):
                # expand these as expected by predict()
                X = np.expand_dims(eval_seq[key][0][row, :(self.n_in), :], axis=0)
                lastknown =  np.expand_dims(np.expand_dims(eval_seq[key][1][row, (self.n_in-1), :], axis=0), axis=0)
                pred = np.expand_dims(np.squeeze(self.predict(X, lastknown)), axis=0)
                gt = eval_seq[key][1][row, (self.n_in):, :]
                rmse = sqrt(mean_squared_error(pred, gt))
                rmse_val.append(rmse)

            if (key == 'Key') or (key == 'Key Train Data'):
                # plt.plot(rmse_val, 'ro', label=key)
                pass
            elif (key == 'Key Held Out'):
                # plt.plot(rmse_val, 'bo', label=key)
                pass
            elif ('Held Out' in key):
                plt.plot(rmse_val, '*', label=key)
            else:
                plt.plot(rmse_val, 'x', label=key)

        plt.title(title)
        plt.ylabel('Mse')
        plt.legend()
        plt.show()
    
    def plot_results (self, history, title=None):
        plt.close()
        fig, ax = plt.subplots()

        for i, hist in enumerate(history):
            if i == 0:
                ax.plot(hist['train'], color='blue', label='train')
                if ('valid' in hist.keys()):
                    ax.plot(hist['valid'], color='orange', label='valid')
            else:
                ax.plot(hist['train'], color='blue')
                if ('valid' in hist.keys()):
                    ax.plot(hist['valid'], color='orange')
                
            print('%d) TrainRMSE=%f' % (i, hist['train'][-1]))
            if ('valid' in hist.keys()):
                  print ('%d) ValidRMSE=%f' % (i, hist['valid'][-1]))
                    
        plt.legend()
        plt.title(title)
        plt.ylabel('RMSE')
        plt.xlabel('N. Epochs')
        plt.show()
        
    def get_segment(self, orig_series, init_idx, end_idx):
        segment = np.full(orig_series.shape, np.nan)
        segment[init_idx:end_idx] = orig_series[init_idx:end_idx]
        return segment 

    def get_date_range(self, dates, orig_dates):
        init = list(orig_dates).index(dates[0])
        end = list(orig_dates).index(dates[-1])
        return init, end

    def plot_results_sequence(self, orig_series, orig_dates, eval_seq_all):
        '''
            eval_seq: dict of tuples - each entry is a sequence to be evaluated. 
            The first entry on tuple is the 'diff' version of the sequence, and 
            the second entry is the 'raw' version of the sequence
            Third entry are the dates
        '''
        assert (orig_series.shape[0] == len(orig_dates))

        plt.figure(figsize=(22,8))
        plt.plot(orig_series, 'b-', label='Series')
        
        for key in eval_seq_all:
            eval_seq = eval_seq_all[key]
            test_dates = eval_seq[2]

            for row in range(eval_seq[0].shape[0]):
                # expand these as expected by predict()
                X = np.expand_dims(eval_seq[0][row, :(self.n_in), :], axis=0) 
                lastknown =  np.expand_dims(np.expand_dims(eval_seq[1][row, (self.n_in-1), :], axis=0), axis=0)
                pred = np.expand_dims(np.squeeze(self.predict(X, lastknown)), axis=0)
                gt = eval_seq[1][row, (self.n_in):, :]
                mse = round(sqrt(mean_squared_error(pred, gt)),2)

                # plotting segment
                init, end = self.get_date_range(test_dates[row], orig_dates)

                if (key == 'Normal'):
                    plt.plot(end, orig_series[end], 'g*')
                else:
                    plt.plot(end, orig_series[end], 'r*')

                plt.annotate('%s'% mse,  xy=(end-1, orig_series[end-1]), weight='bold')

                if (key == 'Anomaly'):
                    plt.title ('Test Normal Sequences')
                else:
                    plt.title ('Test Anomaly Sequences')

        nt = orig_series.shape[0]
        xticks_pos, xticks_labels = get_ticks_and_labels([0, nt-1], 12, 0, orig_dates)
        plt.xticks(xticks_pos, xticks_labels, rotation='vertical')
        plt.xlabel('Time')
        plt.ylabel('Displacement')
        plt.show()