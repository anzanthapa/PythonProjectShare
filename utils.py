# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 19:46:07 2023

@author: anzan
"""
import numpy as np
import pandas as pd
import warnings
from scipy import signal
from scipy import stats
class utils:
    @staticmethod
    def IMAEEG(eeg_data):
        # Function code
        pass
    
    @staticmethod
    def extract_epochs(continuous_signal, event_markers, fs, relative_epoch_start, relative_epoch_end):
        """
        This function provides the segments the continous EEG data and makes epochs based on the 'event_markers'
        provided. it extracts EEG from 'relative_epoch_start' to 'relative_epoch_end' time locked to the event markers. 
        Parameters:
            continous_signal (2D or 1D numpy array): This is the continous signal with size n*nCH where nCh represents number of channels
            and n is the total number of EEG samples.
            event_markers (list): time of the onset of your event in seconds
        """
        
        # Ensure continuous_eeg_signal is a NumPy 2D array
        if not isinstance(continuous_signal, np.ndarray):
            raise ValueError("continuous_signal should be a 2D or 1D NumPy array.")
        if len(continuous_signal.shape)==1:
            continuous_signal_updated = continuous_signal.reshape(-1,1)
        else:
            continuous_signal_updated = continuous_signal
            
        #Validate the event markers
        if isinstance(event_markers, np.ndarray):
            if event_markers.ndim == 1: # Check if 'event_markers' is a 1D numpy array
                event_markers = event_markers.tolist()# Convert to list
            else:
                warnings.warn("'event_markers' is not a 1D numpy array.")
        elif not isinstance(event_markers, list):
            warnings.warn("'event_markers' is not a list.")
        #Validating the fs
        if fs is None:
            fs = 1
        else:
            fs = fs
    
        num_channels = continuous_signal_updated.shape[1]
        time_axis = np.arange(relative_epoch_start * fs, relative_epoch_end * fs) / fs
    
        num_epochs = len(event_markers)
        num_samples_per_epoch = (int((relative_epoch_end - relative_epoch_start) * fs))
        epochs_data = np.zeros((num_epochs, num_samples_per_epoch, num_channels))
    
        for i in range(num_epochs):
            start_sample = int(round(event_markers[i] * fs) + round(relative_epoch_start * fs))
            end_sample = int(round(event_markers[i] * fs) + round(relative_epoch_end * fs)-1)
    
            if start_sample >= 0 and end_sample < continuous_signal_updated.shape[0]:
                epoch_data = continuous_signal_updated[start_sample:end_sample + 1, :]
                epochs_data[i, :, :] = epoch_data
            else:
                print('There are not enough samples within the range.')
    
        return epochs_data, time_axis
    
    @staticmethod
    def compute_features(eeg_epoch: np.ndarray,fs: int, feature_type: str):
        # Check if eeg_epoch is an instance of np.ndarray
        if not isinstance(eeg_epoch, np.ndarray):
            raise TypeError(f"Expected eeg_epoch to be a numpy.ndarray, got {type(eeg_epoch).__name__}")

        # Check dimensionality and reshape if necessary
        if eeg_epoch.ndim == 1:
            eeg_epoch = eeg_epoch.reshape(-1, 1)  # Reshape 1D to 2D with 1 channel
        elif eeg_epoch.ndim != 2:
            raise ValueError(f"Expected eeg_epoch to be 1D or 2D array, got {eeg_epoch.ndim}-dimensional array")
        
        nch = eeg_epoch.shape[1]
        #### FEATURE COMPUTATION
        if feature_type.lower() == 'erp':
            feature_matrix = eeg_epoch
        elif feature_type.lower() == 'psd':
            feature_matrix = np.empty((5,nch))
            for chi in range(nch):
                freq, psd = signal.welch(eeg_epoch[:,chi],fs=fs,nperseg = fs/2,noverlap = None)
                delta_power = np.sum(psd[(freq>=0) & (freq<=4)])
                theta_power = np.sum(psd[(freq>4) & (freq<=8)])
                alpha_power = np.sum(psd[(freq>8) & (freq<=13)])
                beta_power = np.sum(psd[(freq>13) & (freq<=30)])
                gamma_power = np.sum(psd[(freq>30)])
                feature_matrix[:,chi] = [delta_power,theta_power,alpha_power,beta_power,gamma_power]
                # feature_matrix[:,chi] = psd
        elif feature_type.lower() == 'autocorr':
            feature_matrix = np.empty(eeg_epoch.shape)
            for chi in range(nch):
                signal_for_autocorr = eeg_epoch[:,chi]-eeg_epoch[:,chi].mean()
                auto_corr = signal.correlate(signal_for_autocorr, signal_for_autocorr,method = 'direct')
                feature_matrix[:,chi] = auto_corr[eeg_epoch.shape[0]-1:]/(np.var(eeg_epoch[:,chi])*len(eeg_epoch[:,chi]))
                del auto_corr, signal_for_autocorr
        elif feature_type.lower() == 'entropy':
            feature_matrix = np.empty((1,eeg_epoch.shape[1]))
            for chi in range(nch):
                entropy = stats.entropy(eeg_epoch[:,chi])
                feature_matrix[0,chi] = entropy
        else:
            raise ValueError("Unsupported feature type specified")
        return feature_matrix
    
    @staticmethod
    def extract_features(eeg_epoch_raw,fs,ch_names,include_erp = True, include_PSD = True, include_AC = True, include_LPFERP = True):
        
        ear_channel_indices = [ch_names.index('A1'),ch_names.index('A2')]
        
        
        average_mastoid = np.mean(eeg_epoch_raw[:,ear_channel_indices],axis = 1).reshape(-1,1)
        eeg_epoch_AM = eeg_epoch_raw - average_mastoid
        eeg_epoch = eeg_epoch_AM
        
        features = []
        if include_erp:
            channels_for_ERP = ['F3','F4','C3','C4','P3','P4','F7','T3','T4','Fz','Cz','Pz']
            channel_index_for_ERP = [ch_names.index(ch) for ch in channels_for_ERP if ch in ch_names]
            ERP_features = utils.compute_features(eeg_epoch, fs, 'erp')
            ERP = ERP_features[:,channel_index_for_ERP].flatten().reshape(1,-1)
            features.append(ERP)
        if include_PSD: 
            channel_for_BP = ['FP1','Fp2','F3','F4','C3','C4','P3','P4','O1','02','F7','T3','T4','Fz','Cz','Pz']
            channel_index_for_BP = [ch_names.index(ch) for ch in channel_for_BP if ch in ch_names]
            BP_features = utils.compute_features(eeg_epoch, fs, 'psd')
            BP = BP_features[:,channel_index_for_BP].flatten().reshape(1,-1)
            features.append(BP)
        if include_AC: 
            channel_for_AC = ['FP1','Fp2','F4','C3','C4','P3','P4','O1','02','F7','F8','T3','T5','Fz','Cz','Pz']
            channel_index_for_AC = [ch_names.index(ch) for ch in channel_for_AC if ch in ch_names]
            AC_features = utils.compute_features(eeg_epoch, fs, 'autocorr')
            AC = AC_features[:,channel_index_for_AC].flatten().reshape(1,-1)
            features.append(AC)
        if include_LPFERP:
            channels_for_LPFERP = ['F3','F4','C3','C4','P3','P4','F7','T3','T4','Fz','Cz','Pz']
            channel_index_for_LPFERP = [ch_names.index(ch) for ch in channels_for_LPFERP if ch in ch_names]
            
            sos = signal.ellip(13,0.01,60,Wn=8,btype = 'lowpass',fs=fs,output = 'sos')
            eeg_data_lpfiltered = np.zeros((eeg_epoch.shape[0],len(channel_index_for_LPFERP)))
            for i,chi in enumerate(channel_index_for_LPFERP):
                eeg_data_lpfiltered_temp = signal.sosfiltfilt(sos,eeg_epoch[:,chi])
                eeg_data_lpfiltered[:,i] = eeg_data_lpfiltered_temp
                del eeg_data_lpfiltered_temp
                
            LPFERP = eeg_data_lpfiltered.flatten().reshape(1,-1)
            features.append(LPFERP)        
            
        return np.hstack(features)
    
    @staticmethod
    def importdata(file_path,delimiter=None,header=None):
        """
        Import and process data from a .dat or .csv file.

        Parameters:
        -----------
        file_path : str
            Path to the file to be imported (e.g., .dat file).
        delimiter : str, optional
            Delimiter used in the file (default is 'None').
        header : int, None, or str, optional
            Row number(s) to use as the column names. Default is None (no header).

        Returns:
        --------
        data : np.ndarray
            2D NumPy array containing the data i.e (channels,samples).
            rows represent channels
            columns represents data sample
        channel_names : np.ndarray
            Array of channel names.
            
        Note: 'channel_names' have the same size as the channels in 'data'
        
        Raises:
        -------
        FileNotFoundError:
            If the file at file_path does not exist.
        ValueError:
            If the file cannot be properly parsed or contains invalid data.
        """
        
        try:
            # Read the file into a DataFrame using pandas
            df = pd.read_csv(file_path,delimiter=delimiter, header = header)# Adjust delimiter as needed
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except pd.errors.ParserError as pe:
            raise ValueError(f"Error parsing file: {pe}")
        except Exception as e:
            print(f"An error occurred: {e}")
            
        # Extract number of channels (rows)
        channel_num = df.shape[0]
        channel_names_list = []
        data_list = []
        # Iterate over each row to extract channel names and data
        for channeli in range(channel_num):
            elements = df.iloc[channeli,0].strip('[]').split()
            channel_names_list.append(elements[0])
            data_list.append(elements[1:])
            
        # Convert data_list to a NumPy array and ensure correct dtype
        try:
            data = np.vstack(data_list).astype(float)
        except ValueError as ve:
            raise ValueError(f"Data conversion error: {ve}")
        # Convert channel names list to a NumPy array
        channel_names = np.array(channel_names_list, dtype=str)
        return  data, channel_names

        

