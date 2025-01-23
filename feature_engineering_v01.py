# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:07:35 2023

@author: anzan
"""
#%% Clearing the variables and initilizing the timer 
from IPython import get_ipython   
try:
    get_ipython().run_line_magic('reset', '-sf')
except AttributeError:
    print("Not running in an IPython environment. Skipping reset.")

import time
script_start = time.time()

#%% Importing Required Libraries 
import os
# Changing the current directory to the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir));
# Importing additional libraries
import numpy as np
from utils import utils
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from itertools import product
import pandas as pd
#%% Extracting EDF Files & Human Score Files from respective directories
dataset_path = os.path.join(parent_dir, "data\\raw")
result_path = os.path.join(parent_dir, "results")
os.makedirs(result_path) if not os.path.exists(result_path) else None
os.makedirs(os.path.join(parent_dir, "data\\processed")) if not os.path.exists(result_path) else None
# Filter out only the files from the list
raw_files = [file for file in os.listdir(dataset_path) if file.endswith(".mat")]
print(f'The raw files are as follows: \n{raw_files}')
#%% DATA Processing & VISUALIZATION
start_timer_sec1 = time.time()
class_info_all_list = []
EEGEpochs_all_list = []
EEGEpochs_AM_all_list = []
EEGEpochs_AM_LPF_all_list = []
#%% Starting the loop
for filei in range(0,len(raw_files)): #len(raw_files)
    #%% Reading the file
    current_file_name = raw_files[filei]
    print(f'The file "{current_file_name}" is being processed...')
    raw_data_struct = scipy.io.loadmat(os.path.join(dataset_path,current_file_name))['o']
    os.makedirs(result_path+'\\' +current_file_name) if not os.path.exists(result_path+'\\' +current_file_name) else None
    #%%% Getting Basic Information
    eeg_data = raw_data_struct['data'].item()
    event_markers = raw_data_struct['marker'].item()
    ch_names = [item[0][0] for item in raw_data_struct['chnames'].item()]
    fs = raw_data_struct['sampFreq'][0][0].item()
    bad_indices = [ch_names.index('A1'),ch_names.index('A2'),ch_names.index('X5')]
    ear_channel_indices = [ch_names.index('A1'),ch_names.index('A2')]
    valid_channel_indices = [i for i in range(0, len(ch_names)-1) if i not in bad_indices]
    print(bad_indices)
    # #Plotting event markers
    # time_axis_full = np.arange(start=0, stop=len(event_markers)/fs, step=1/fs)
    # event_markers[event_markers == 99] =0
    # plt.figure()
    # plt.plot(time_axis_full,event_markers)
    #%%% Determining the class info & EPOCHS
    class_info_raw = np.empty((0,4))
    for isample in range(0,eeg_data.shape[0]-1):
        current_class_info = np.full((1, 4), np.nan)
        if(event_markers[isample][0]==0 and (event_markers[isample+1][0]==1 or event_markers[isample+1][0]==2) ):
            end_index = np.where(np.diff(event_markers[isample:int(isample+1.5*fs),0].astype(np.float16)) < 0)[0][0]+1
            current_class_info = np.array([event_markers[isample+1][0],isample+1,isample+end_index,filei])
            class_info_raw = np.vstack((class_info_raw,current_class_info))
            del current_class_info

    epoch_start = -1.5
    epoch_end = 0.75
    trial_gaps =(class_info_raw[1:,1]-class_info_raw[0:-1,2])/fs
    print(f'Average inter-trial period is {np.mean(trial_gaps)} seconds for {class_info_raw.shape[0]} trials.')
    within_trial_gap = (class_info_raw[0:,2]-class_info_raw[0:,1])/fs
    print(f'Average Trial Period is {np.mean(within_trial_gap)} seconds for {class_info_raw.shape[0]} trials.')
    valid_trials = np.where(trial_gaps>abs(epoch_start))[0]+1
    valid_trials = np.insert(valid_trials,0,0)
    class_info = class_info_raw[valid_trials,:]
    print(f'{class_info.shape[0]}/{class_info_raw.shape[0]} were used for further analysis.')
    
    [EEGEpochs,time_axis] =  utils.extract_epochs(eeg_data, (class_info[:,1]/fs).tolist(), fs, epoch_start, epoch_end)
    rest_index = np.where((time_axis>=-1.5) & (time_axis<=-0.75))[0]
    premov_index = np.where((time_axis>=-0.75) & (time_axis<=0))[0]
    # Find all occurrences of 1 & 2
    class1_index = [i for i, x in enumerate(class_info[:,0]) if x == 1]
    class2_index = [i for i, x in enumerate(class_info[:,0]) if x == 2]
    #%% RAW FEATURE EXTRACTION
    #ERP 
    raw_ERP = np.stack([utils.compute_features(EEGEpochs[i], fs, 'erp') for i in range(EEGEpochs.shape[0])])
    raw_ERP_class1 = raw_ERP[class1_index,:,:]
    raw_ERP_class2 = raw_ERP[class2_index,:,:]
    #%%% Plotting ERP disregarding the class
    # Create a PdfPages object to save figures in a PDF file
    pdf_pages_ERP = PdfPages(os.path.join(result_path,current_file_name,'averageERP_raw_1vs2.pdf'))
    for chi in range(0,22):
        channelEpochs_class1 = raw_ERP_class1[:,:,chi]
        channelEpochs_class2 = raw_ERP_class2[:,:,chi]
        currentChannel = ch_names[chi]
        # Calculate the mean and standard deviation along axis 0 (across all rows)
        averageERP_together = np.mean(EEGEpochs[:,:,chi],axis = 0)
        stdERP_together = np.std(EEGEpochs[:,:,chi],axis = 0)
        averageERP_class1 = np.mean(channelEpochs_class1, axis=0)
        stdERP_class1 = np.std(channelEpochs_class1, axis=0)
        averageERP_class2 = np.mean(channelEpochs_class2, axis=0)
        stdERP_class2 = np.std(channelEpochs_class2, axis=0)
        fig,axes = plt.subplots(figsize=(10,6))
        plt.plot(time_axis,averageERP_class1,'m-',label = 'D Key Press',linewidth = 2.5)
        plt.plot(time_axis,averageERP_class1+stdERP_class1,'m--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_class1-stdERP_class1,'m--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_class2,'c-',label = 'L Key Press',linewidth = 2.5)
        plt.plot(time_axis,averageERP_class2+stdERP_class2,'c--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_class2-stdERP_class2,'c--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_together,'k-',label = 'Mean of Both Cases',linewidth = 1.5)
        plt.plot(time_axis,averageERP_together+stdERP_together,'k--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_together-stdERP_together,'k--',label = '',linewidth = 1)
        plt.title('Channel Name: ' + ch_names[chi])
        plt.xlabel('Time relative to movement onset[sec]')
        plt.ylabel('Amplitude[uV]')
        plt.axvline(-0.75,color = 'black', linestyle='--')
        plt.axvline(0,color = 'black', linestyle='--')
        plt.legend()
        # Save the plot as a PDF file
        pdf_pages_ERP.savefig(fig)
        plt.close()
    pdf_pages_ERP.close()
    
    #%% Plotting Band Powers in 5 different bands of raw EEG

    raw_BP_premov = np.stack([utils.compute_features(EEGEpochs[i,premov_index,:], fs, 'psd') for i in range(EEGEpochs.shape[0])])
    raw_BP_rest = np.stack([utils.compute_features(EEGEpochs[i,rest_index,:], fs, 'psd') for i in range(EEGEpochs.shape[0])])
    
    average_BP_premov = np.mean(raw_BP_premov,axis = 0)
    average_BP_rest = np.mean(raw_BP_rest,axis = 0)
    
    std_BP_premov = np.std(raw_BP_premov,axis = 0)
    std_BP_rest = np.std(raw_BP_rest,axis = 0)
    frequency_bands = ['Delta[0Hz-4Hz]', 'Theta[4Hz-8Hz]', 'Alpha[8Hz-13Hz]', 'Beta[13Hz-30Hz]','Gamma[30Hz-100Hz]']
    time_frames = ['[-1.5s -0.75s]', '[-0.75s 0s]']
    
    pdf_pages_BP = PdfPages(os.path.join(result_path,current_file_name,'averageBP_raw.pdf'))
    for chi in range(0,22):
        currentChannel = ch_names[chi]
        # Number of frequency bands
        n_bands = len(frequency_bands)
        # Setting up the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot each frequency band
        ax.plot(average_BP_premov[:,chi],color = 'g',linestyle='--',marker = '8',markersize = 10,label='Premovement Stage')
        ax.plot(average_BP_rest[:,chi],color = 'r',linestyle='--',marker = '8',markersize = 10,label='Resting Stage')
        # Adding labels and title
        ax.set_xlabel('Frequency Bands', fontsize=12)
        ax.set_ylabel('Power[V^2/Hz]', fontsize=12)
        ax.set_title(f'Frequency Band Power across Different Movement Phases for {ch_names[chi]}', fontsize=14)
        # Setting the x-ticks to be in the middle of the group of bars
        ax.set_xticks(np.arange(n_bands))
        ax.set_xticklabels(frequency_bands)
        ax.set_xlim(-0.75,4.75)
        # Adding a legend
        ax.legend(title='Movement Phases:')
        # Save the plot as a PDF file
        pdf_pages_BP.savefig(fig)
        # Closing the plot
        plt.close()
    pdf_pages_BP.close()
    del average_BP_premov, average_BP_rest, std_BP_premov, std_BP_rest
    #%% Average Mastoid Re-referencing
    average_mastoid = np.mean(eeg_data[:,ear_channel_indices],axis = 1).reshape(-1,1)
    eeg_data_processed1 = eeg_data - average_mastoid

    [EEGEpochs_AM,time_axis] = utils.extract_epochs(eeg_data_processed1, (class_info[:,1]/fs).tolist(), fs, epoch_start, epoch_end)
    AM_ERP = np.stack([utils.compute_features(EEGEpochs_AM[i], fs, 'erp') for i in range(EEGEpochs.shape[0])])
    AM_ERP_class1 = AM_ERP[class1_index,:,:]
    AM_ERP_class2 = AM_ERP[class2_index,:,:]
    
    #%%% Plotting ERP after re-referencing
    # Create a PdfPages object to save figures in a PDF file
    pdf_pages_ERP = PdfPages(os.path.join(result_path,current_file_name,'averageERP_AMastoid_1vs2.pdf'))
    for chi in range(0,22):
        channelEpochs_class1 = AM_ERP_class1[:,:,chi]
        channelEpochs_class2 = AM_ERP_class2[:,:,chi]
        currentChannel = ch_names[chi]
        # Calculate the mean and standard deviation along axis 0 (across all rows)
        averageERP_together = np.mean(EEGEpochs_AM[:,:,chi],axis = 0)
        stdERP_together = np.std(EEGEpochs_AM[:,:,chi],axis = 0)
        averageERP_class1 = np.mean(channelEpochs_class1, axis=0)
        stdERP_class1 = np.std(channelEpochs_class1, axis=0)
        averageERP_class2 = np.mean(channelEpochs_class2, axis=0)
        stdERP_class2 = np.std(channelEpochs_class2, axis=0)
        fig,axes = plt.subplots(figsize=(10,6))
        plt.plot(time_axis,averageERP_class1,'m-',label = 'D Key Press',linewidth = 2.5)
        plt.plot(time_axis,averageERP_class1+stdERP_class1,'m--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_class1-stdERP_class1,'m--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_class2,'c-',label = 'L Key Press',linewidth = 2.5)
        plt.plot(time_axis,averageERP_class2+stdERP_class2,'c--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_class2-stdERP_class2,'c--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_together,'k-',label = 'Mean of Both Cases',linewidth = 1.5)
        plt.plot(time_axis,averageERP_together+stdERP_together,'k--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_together-stdERP_together,'k--',label = '',linewidth = 1)
        plt.title('Channel Name: ' + ch_names[chi])
        plt.xlabel('Time relative to movement onset[sec]')
        plt.ylabel('Amplitude[uV]')
        plt.axvline(-0.75,color = 'black', linestyle='--')
        plt.axvline(0,color = 'black', linestyle='--')
        plt.legend()
        # Save the plot as a PDF file
        pdf_pages_ERP.savefig(fig)
        plt.close()
    pdf_pages_ERP.close()
    
    #%% Plotting Band Powers in 5 different bands of AMastoid EEG 
    AM_BP_premov = np.stack([utils.compute_features(EEGEpochs_AM[i,premov_index,:], fs, 'psd') for i in range(EEGEpochs_AM.shape[0])])
    AM_BP_rest = np.stack([utils.compute_features(EEGEpochs_AM[i,rest_index,:], fs, 'psd') for i in range(EEGEpochs_AM.shape[0])])
    
    average_BP_premov = np.mean(AM_BP_premov,axis = 0)
    average_BP_rest = np.mean(AM_BP_rest,axis = 0)
    
    std_BP_premov = np.std(AM_BP_premov,axis = 0)
    std_BP_rest = np.std(AM_BP_rest,axis = 0)
    frequency_bands = ['Delta[0Hz-4Hz]', 'Theta[4Hz-8Hz]', 'Alpha[8Hz-13Hz]', 'Beta[13Hz-30Hz]','Gamma[30Hz-100Hz]']
    time_frames = ['[-1.5s -0.75s]', '[-0.75s 0s]']
    
    pdf_pages_BP = PdfPages(os.path.join(result_path,current_file_name,'averageBP_AMastoid.pdf'))
    for chi in range(0,22):
        currentChannel = ch_names[chi]
        # Number of frequency bands
        n_bands = len(frequency_bands)
        # Setting up the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot each frequency band
        ax.plot(average_BP_premov[:,chi],color = 'g',linestyle='--',marker = '8',markersize = 10,label='Premovement Stage')
        ax.plot(average_BP_rest[:,chi],color = 'r',linestyle='--',marker = '8',markersize = 10,label='Resting Stage')
        # Adding labels and title
        ax.set_xlabel('Frequency Bands', fontsize=12)
        ax.set_ylabel('Power[V^2/Hz]', fontsize=12)
        ax.set_title(f'Frequency Band Power across Different Movement Phases for {ch_names[chi]}', fontsize=14)
        # Setting the x-ticks to be in the middle of the group of bars
        ax.set_xticks(np.arange(n_bands))
        ax.set_xticklabels(frequency_bands)
        ax.set_xlim(-0.75,4.75)
        # Adding a legend
        ax.legend(title='Movement Phases:')
        # Save the plot as a PDF file
        pdf_pages_BP.savefig(fig)
        # Closing the plot
        plt.close()
    pdf_pages_BP.close()
    del average_BP_premov, average_BP_rest, std_BP_premov, std_BP_rest
    #%% LOW PASS FILTER DESIGN (4Hz Cutoff)
    sos = signal.ellip(13,0.01,60,Wn=8,btype = 'lowpass',fs=fs,output = 'sos')
    w,h = signal.sosfreqz(sos,worN =512,fs = fs)
    # plt.subplot(2, 1, 1)
    # db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    # plt.plot(w, db)
    # plt.ylim(-75, 5)
    # plt.xlim(0,100)
    # plt.grid(True)
    # plt.yticks([0, -20, -40, -60])
    # plt.ylabel('Gain [dB]')
    # plt.title('Frequency Response')
    # plt.subplot(2, 1, 2)
    # plt.plot(w, np.angle(h))
    # plt.grid(True)
    # plt.xlim(0,100)
    # plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
    #             [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    # plt.ylabel('Phase [rad]')
    # plt.xlabel(' Frequency (Hz)')
    # plt.show()
    eeg_data_lpfiltered = np.zeros(eeg_data.shape)
    for chi in range(len(ch_names)):
        eeg_data_lpfiltered_temp = signal.sosfiltfilt(sos,eeg_data_processed1[:,chi])
        eeg_data_lpfiltered[:,chi] = eeg_data_lpfiltered_temp
        del eeg_data_lpfiltered_temp
    
    [EEGEpochs_AM_LPF,time_axis] = utils.extract_epochs(eeg_data_lpfiltered, (class_info[:,1]/fs).tolist(), fs, epoch_start, epoch_end)
    AM_LPF_ERP = np.stack([utils.compute_features(EEGEpochs_AM_LPF[i], fs, 'erp') for i in range(EEGEpochs.shape[0])])
    
    #% Plotting ERP after re-referencing and LPF
    # Create a PdfPages object to save figures in a PDF file
    pdf_pages_ERP = PdfPages(os.path.join(result_path,current_file_name,'averageERP_AMastoid_LPF_1vs2.pdf'))
    for chi in range(0,22):
        channelEpochs_class1 = np.squeeze(AM_LPF_ERP[np.where(class_info[:,0]==1),:,chi])
        channelEpochs_class2 = np.squeeze(AM_LPF_ERP[np.where(class_info[:,0]==2),:,chi])
        currentChannel = ch_names[chi]
        # Calculate the mean and standard deviation along axis 0 (across all rows)
        averageERP_together = np.mean(AM_LPF_ERP[:,:,chi],axis = 0)
        stdERP_together = np.std(AM_LPF_ERP[:,:,chi],axis = 0)
        averageERP_class1 = np.mean(channelEpochs_class1, axis=0)
        stdERP_class1 = np.std(channelEpochs_class1, axis=0)
        averageERP_class2 = np.mean(channelEpochs_class2, axis=0)
        stdERP_class2 = np.std(channelEpochs_class2, axis=0)
        fig,axes = plt.subplots(figsize=(10,6))
        plt.plot(time_axis,averageERP_class1,'m-',label = 'D Key Press',linewidth = 2.5)
        plt.plot(time_axis,averageERP_class1+stdERP_class1,'m--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_class1-stdERP_class1,'m--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_class2,'c-',label = 'L Key Press',linewidth = 2.5)
        plt.plot(time_axis,averageERP_class2+stdERP_class2,'c--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_class2-stdERP_class2,'c--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_together,'k-',label = 'Mean of Both Cases',linewidth = 1.5)
        plt.plot(time_axis,averageERP_together+stdERP_together,'k--',label = '',linewidth = 1)
        plt.plot(time_axis,averageERP_together-stdERP_together,'k--',label = '',linewidth = 1)
        plt.title('Channel Name: ' + ch_names[chi])
        plt.xlabel('Time relative to movement onset[sec]')
        plt.ylabel('Amplitude[uV]')
        plt.axvline(-0.75,color = 'black', linestyle='--')
        plt.axvline(0,color = 'black', linestyle='--')
        plt.legend()
        # Save the plot as a PDF file
        pdf_pages_ERP.savefig(fig)
        plt.close()
    pdf_pages_ERP.close()    
    #%% AUTOCORRELATION after Average Mastoids
    AM_AC_premov = np.stack([utils.compute_features(EEGEpochs_AM[i,premov_index,:], fs, 'autocorr') for i in range(EEGEpochs_AM.shape[0])])
    AM_AC_rest = np.stack([utils.compute_features(EEGEpochs_AM[i,rest_index,:], fs, 'autocorr') for i in range(EEGEpochs_AM.shape[0])])
    
    average_AM_AC_premov = np.mean(AM_AC_premov,axis = 0)
    average_AM_AC_rest = np.mean(AM_AC_rest,axis = 0)
    
    std_AM_AC_premov = np.std(AM_AC_premov,axis = 0)
    std_AM_AC_rest = np.std(AM_AC_rest,axis = 0)
    
    pdf_pages_AC = PdfPages(os.path.join(result_path,current_file_name,'averageAC_AMastoid.pdf'))
    for chi in range(0,22):
        currentChannel = ch_names[chi]
        # Number of frequency bands
        n_bands = len(frequency_bands)
        # Setting up the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot each frequency band
        ax.plot(rest_index/fs,average_AM_AC_premov[:,chi],color = 'g',linestyle='-',label='Premovement Stage')
        ax.plot(rest_index/fs,average_AM_AC_rest[:,chi],color = 'r',linestyle='-',label='Resting Stage')
        # Adding labels and title
        ax.set_xlabel('Lags[secs]', fontsize=12)
        ax.set_ylabel('AutoCorrelation', fontsize=12)
        ax.set_title(f'Autocorrealtion for {ch_names[chi]}', fontsize=14)
        # Adding a legend
        ax.legend(title='Movement Phases:')
        # Save the plot as a PDF file
        pdf_pages_AC.savefig(fig)
        # Closing the plot
        plt.close()
    pdf_pages_AC.close()
    del average_AM_AC_premov, average_AM_AC_rest, std_AM_AC_premov, std_AM_AC_rest
    #%% AUTOCORRELATION after Average Mastoids and LPF
    AM_LPF_AC_premov = np.stack([utils.compute_features(EEGEpochs_AM_LPF[i,premov_index,:], fs, 'autocorr') for i in range(EEGEpochs_AM_LPF.shape[0])])
    AM_LPF_AC_rest = np.stack([utils.compute_features(EEGEpochs_AM_LPF[i,rest_index,:], fs, 'autocorr') for i in range(EEGEpochs_AM_LPF.shape[0])])
    
    average_AM_LPF_AC_premov = np.mean(AM_LPF_AC_premov,axis = 0)
    average_AM_LPF_AC_rest = np.mean(AM_LPF_AC_rest,axis = 0)
    
    std_AM_LPF_AC_premov = np.std(AM_LPF_AC_premov,axis = 0)
    std_AM_LPF_AC_rest = np.std(AM_LPF_AC_rest,axis = 0)
    
    pdf_pages_AC = PdfPages(os.path.join(result_path,current_file_name,'averageAC_AMastoid_LPF.pdf'))
    for chi in range(0,22):
        currentChannel = ch_names[chi]
        # Number of frequency bands
        n_bands = len(frequency_bands)
        # Setting up the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot each frequency band
        ax.plot(rest_index/fs,average_AM_LPF_AC_premov[:,chi],color = 'g',linestyle='-',label='Premovement Stage')
        ax.plot(rest_index/fs,average_AM_LPF_AC_rest[:,chi],color = 'r',linestyle='-',label='Resting Stage')
        # Adding labels and title
        ax.set_xlabel('Lags[secs]', fontsize=12)
        ax.set_ylabel('AutoCorrelation', fontsize=12)
        ax.set_title(f'Autocorrealtion for {ch_names[chi]}', fontsize=14)
        # Adding a legend
        ax.legend(title='Movement Phases:')
        # Save the plot as a PDF file
        pdf_pages_AC.savefig(fig)
        # Closing the plot
        plt.close()
    pdf_pages_AC.close()
    del average_AM_LPF_AC_premov, average_AM_LPF_AC_rest, std_AM_LPF_AC_premov, std_AM_LPF_AC_rest   
    #%% Appending Epochs & Time Axis
    if filei != 2:
        class_info_all_list.append(class_info)
        EEGEpochs_all_list.append(EEGEpochs)
        EEGEpochs_AM_all_list.append(EEGEpochs_AM)
        EEGEpochs_AM_LPF_all_list.append(EEGEpochs_AM_LPF)
    print(f'The file "{current_file_name}" is done processing.')
print('Combining Epochs is done.')
#%% Creating Array of Both Subjects for further analysis all together
class_info_all = np.vstack(class_info_all_list)
EEGEpochs_all = np.vstack(EEGEpochs_all_list)
EEGEpochs_AM_all = np.vstack(EEGEpochs_AM_all_list)
EEGEpochs_AM_LPF_all = np.vstack(EEGEpochs_AM_LPF_all_list)
del class_info_all_list, EEGEpochs_all_list,EEGEpochs_AM_all_list

#%% 
os.makedirs(os.path.join(result_path,'overall','feature_visualization')) if not os.path.exists(os.path.join(result_path,'overall','feature_visualization')) else None
pdf_pages_ERP = PdfPages(os.path.join(result_path,'overall','feature_visualization','averageERP_raw_1vs2.pdf'))
for chi in range(0,22):
    channelEpochs_class1 = np.squeeze(EEGEpochs_all[np.where(class_info_all[:,0]==1),:,chi])
    channelEpochs_class2 = np.squeeze(EEGEpochs_all[np.where(class_info_all[:,0]==2),:,chi])
    currentChannel = ch_names[chi]
    # Calculate the mean and standard deviation along axis 0 (across all rows)
    averageERP_together = np.mean(EEGEpochs_all[:,:,chi],axis = 0)
    stdERP_together = np.std(EEGEpochs_all[:,:,chi],axis = 0)
    averageERP_class1 = np.mean(channelEpochs_class1, axis=0)
    stdERP_class1 = np.std(channelEpochs_class1, axis=0)
    averageERP_class2 = np.mean(channelEpochs_class2, axis=0)
    stdERP_class2 = np.std(channelEpochs_class2, axis=0)
    fig,axes = plt.subplots(figsize=(10,6))
    plt.plot(time_axis,averageERP_class1,'m-',label = 'D Key Press',linewidth = 2.5)
    plt.plot(time_axis,averageERP_class1+stdERP_class1,'m--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_class1-stdERP_class1,'m--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_class2,'c-',label = 'L Key Press',linewidth = 2.5)
    plt.plot(time_axis,averageERP_class2+stdERP_class2,'c--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_class2-stdERP_class2,'c--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_together,'k-',label = 'Mean of Both Cases',linewidth = 1.5)
    plt.plot(time_axis,averageERP_together+stdERP_together,'k--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_together-stdERP_together,'k--',label = '',linewidth = 1)
    plt.title('Channel Name: ' + ch_names[chi])
    plt.xlabel('Time relative to movement onset[sec]')
    plt.ylabel('Amplitude[uV]')
    plt.axvline(-0.75,color = 'black', linestyle='--')
    plt.axvline(0,color = 'black', linestyle='--')
    plt.legend()
    # Save the plot as a PDF file
    pdf_pages_ERP.savefig(fig)
    plt.close()
pdf_pages_ERP.close()

pdf_pages_ERP = PdfPages(os.path.join(result_path,'overall','feature_visualization','averageERP_AMastoid_1vs2.pdf'))
for chi in range(0,22):
    channelEpochs_class1 = np.squeeze(EEGEpochs_AM_all[np.where(class_info_all[:,0]==1),:,chi])
    channelEpochs_class2 = np.squeeze(EEGEpochs_AM_all[np.where(class_info_all[:,0]==2),:,chi])
    currentChannel = ch_names[chi]
    # Calculate the mean and standard deviation along axis 0 (across all rows)
    averageERP_together = np.mean(EEGEpochs_AM_all[:,:,chi],axis = 0)
    stdERP_together = np.std(EEGEpochs_AM_all[:,:,chi],axis = 0)
    averageERP_class1 = np.mean(channelEpochs_class1, axis=0)
    stdERP_class1 = np.std(channelEpochs_class1, axis=0)
    averageERP_class2 = np.mean(channelEpochs_class2, axis=0)
    stdERP_class2 = np.std(channelEpochs_class2, axis=0)
    fig,axes = plt.subplots(figsize=(10,6))
    plt.plot(time_axis,averageERP_class1,'m-',label = 'D Key Press',linewidth = 2.5)
    plt.plot(time_axis,averageERP_class1+stdERP_class1,'m--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_class1-stdERP_class1,'m--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_class2,'c-',label = 'L Key Press',linewidth = 2.5)
    plt.plot(time_axis,averageERP_class2+stdERP_class2,'c--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_class2-stdERP_class2,'c--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_together,'k-',label = 'Mean of Both Cases',linewidth = 1.5)
    plt.plot(time_axis,averageERP_together+stdERP_together,'k--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_together-stdERP_together,'k--',label = '',linewidth = 1)
    plt.title('Channel Name: ' + ch_names[chi])
    plt.xlabel('Time relative to movement onset[sec]')
    plt.ylabel('Amplitude[uV]')
    plt.axvline(-0.75,color = 'black', linestyle='--')
    plt.axvline(0,color = 'black', linestyle='--')
    plt.legend()
    # Save the plot as a PDF file
    pdf_pages_ERP.savefig(fig)
    plt.close()
pdf_pages_ERP.close()

pdf_pages_ERP = PdfPages(os.path.join(result_path,'overall','feature_visualization','averageERP_AMastoid_LPF_1vs2.pdf'))
for chi in range(0,22):
    channelEpochs_class1 = np.squeeze(EEGEpochs_AM_LPF_all[np.where(class_info_all[:,0]==1),:,chi])
    channelEpochs_class2 = np.squeeze(EEGEpochs_AM_LPF_all[np.where(class_info_all[:,0]==2),:,chi])
    currentChannel = ch_names[chi]
    # Calculate the mean and standard deviation along axis 0 (across all rows)
    averageERP_together = np.mean(EEGEpochs_AM_LPF_all[:,:,chi],axis = 0)
    stdERP_together = np.std(EEGEpochs_AM_LPF_all[:,:,chi],axis = 0)
    averageERP_class1 = np.mean(channelEpochs_class1, axis=0)
    stdERP_class1 = np.std(channelEpochs_class1, axis=0)
    averageERP_class2 = np.mean(channelEpochs_class2, axis=0)
    stdERP_class2 = np.std(channelEpochs_class2, axis=0)
    fig,axes = plt.subplots(figsize=(10,6))
    plt.plot(time_axis,averageERP_class1,'m-',label = 'D Key Press',linewidth = 2.5)
    plt.plot(time_axis,averageERP_class1+stdERP_class1,'m--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_class1-stdERP_class1,'m--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_class2,'c-',label = 'L Key Press',linewidth = 2.5)
    plt.plot(time_axis,averageERP_class2+stdERP_class2,'c--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_class2-stdERP_class2,'c--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_together,'k-',label = 'Mean of Both Cases',linewidth = 1.5)
    plt.plot(time_axis,averageERP_together+stdERP_together,'k--',label = '',linewidth = 1)
    plt.plot(time_axis,averageERP_together-stdERP_together,'k--',label = '',linewidth = 1)
    plt.title('Channel Name: ' + ch_names[chi])
    plt.xlabel('Time relative to movement onset[sec]')
    plt.ylabel('Amplitude[uV]')
    plt.axvline(-0.75,color = 'black', linestyle='--')
    plt.axvline(0,color = 'black', linestyle='--')
    plt.legend()
    # Save the plot as a PDF file
    pdf_pages_ERP.savefig(fig)
    plt.close()
pdf_pages_ERP.close()
#%% Plotting Band Powers in 5 different bands of raw EEG

raw_BP_premov = np.stack([utils.compute_features(EEGEpochs_all[i,premov_index,:], fs, 'psd') for i in range(EEGEpochs_all.shape[0])])
raw_BP_rest = np.stack([utils.compute_features(EEGEpochs_all[i,rest_index,:], fs, 'psd') for i in range(EEGEpochs_all.shape[0])])

average_BP_premov = np.mean(raw_BP_premov,axis = 0)
average_BP_rest = np.mean(raw_BP_rest,axis = 0)

std_BP_premov = np.std(raw_BP_premov,axis = 0)
std_BP_rest = np.std(raw_BP_rest,axis = 0)
frequency_bands = ['Delta[0Hz-4Hz]', 'Theta[4Hz-8Hz]', 'Alpha[8Hz-13Hz]', 'Beta[13Hz-30Hz]','Gamma[30Hz-100Hz]']
time_frames = ['[-1.5s -0.75s]', '[-0.75s 0s]']

pdf_pages_BP = PdfPages(os.path.join(result_path,'overall','feature_visualization','averageBP_raw.pdf'))
for chi in range(0,22):
    currentChannel = ch_names[chi]
    # Number of frequency bands
    n_bands = len(frequency_bands)
    # Setting up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot each frequency band
    ax.plot(average_BP_premov[:,chi],color = 'g',linestyle='--',marker = '8',markersize = 10,label='Premovement Stage')
    ax.plot(average_BP_rest[:,chi],color = 'r',linestyle='--',marker = '8',markersize = 10,label='Resting Stage')
    # Adding labels and title
    ax.set_xlabel('Frequency Bands', fontsize=12)
    ax.set_ylabel('Power[V^2/Hz]', fontsize=12)
    ax.set_title(f'Frequency Band Power across Different Movement Phases for {ch_names[chi]}', fontsize=14)
    # Setting the x-ticks to be in the middle of the group of bars
    ax.set_xticks(np.arange(n_bands))
    ax.set_xticklabels(frequency_bands)
    ax.set_xlim(-0.75,4.75)
    # Adding a legend
    ax.legend(title='Movement Phases:')
    # Save the plot as a PDF file
    pdf_pages_BP.savefig(fig)
    # Closing the plot
    plt.close()
pdf_pages_BP.close()
del average_BP_premov, average_BP_rest, std_BP_premov, std_BP_rest

#% Plotting Band Powers in 5 different bands of AMastoid EEG 
AM_BP_premov = np.stack([utils.compute_features(EEGEpochs_AM_all[i,premov_index,:], fs, 'psd') for i in range(EEGEpochs_AM_all.shape[0])])
AM_BP_rest = np.stack([utils.compute_features(EEGEpochs_AM_all[i,rest_index,:], fs, 'psd') for i in range(EEGEpochs_AM_all.shape[0])])

average_BP_premov = np.mean(AM_BP_premov,axis = 0)
average_BP_rest = np.mean(AM_BP_rest,axis = 0)

std_BP_premov = np.std(AM_BP_premov,axis = 0)
std_BP_rest = np.std(AM_BP_rest,axis = 0)
frequency_bands = ['Delta[0Hz-4Hz]', 'Theta[4Hz-8Hz]', 'Alpha[8Hz-13Hz]', 'Beta[13Hz-30Hz]','Gamma[30Hz-100Hz]']
time_frames = ['[-1.5s -0.75s]', '[-0.75s 0s]']

pdf_pages_BP = PdfPages(os.path.join(result_path,'overall','feature_visualization','averageBP_AMastoid.pdf'))
for chi in range(0,22):
    currentChannel = ch_names[chi]
    # Number of frequency bands
    n_bands = len(frequency_bands)
    # Setting up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot each frequency band
    ax.plot(average_BP_premov[:,chi],color = 'g',linestyle='--',marker = '8',markersize = 10,label='Premovement Stage')
    ax.plot(average_BP_rest[:,chi],color = 'r',linestyle='--',marker = '8',markersize = 10,label='Resting Stage')
    # Adding labels and title
    ax.set_xlabel('Frequency Bands', fontsize=12)
    ax.set_ylabel('Power[V^2/Hz]', fontsize=12)
    ax.set_title(f'Frequency Band Power across Different Movement Phases for {ch_names[chi]}', fontsize=14)
    # Setting the x-ticks to be in the middle of the group of bars
    ax.set_xticks(np.arange(n_bands))
    ax.set_xticklabels(frequency_bands)
    ax.set_xlim(-0.75,4.75)
    # Adding a legend
    ax.legend(title='Movement Phases:')
    # Save the plot as a PDF file
    pdf_pages_BP.savefig(fig)
    # Closing the plot
    plt.close()
pdf_pages_BP.close()
del average_BP_premov, average_BP_rest, std_BP_premov, std_BP_rest
#%% AUTOCORRELATION
#%% AUTOCORRELATION after Average Mastoids
AM_all_AC_premov = np.stack([utils.compute_features(EEGEpochs_AM_all[i,premov_index,:], fs, 'autocorr') for i in range(EEGEpochs_AM_all.shape[0])])
AM_all_AC_rest = np.stack([utils.compute_features(EEGEpochs_AM_all[i,rest_index,:], fs, 'autocorr') for i in range(EEGEpochs_AM_all.shape[0])])

average_AM_LPF_AC_premov = np.mean(AM_all_AC_premov,axis = 0)
average_AM_LPF_AC_rest = np.mean(AM_all_AC_rest,axis = 0)

std_AM_LPF_AC_premov = np.std(AM_LPF_AC_premov,axis = 0)
std_AM_LPF_AC_rest = np.std(AM_LPF_AC_rest,axis = 0)

pdf_pages_AC = PdfPages(os.path.join(result_path,'overall','feature_visualization','averageAC_AMastoid.pdf'))
for chi in range(0,22):
    currentChannel = ch_names[chi]
    # Number of frequency bands
    n_bands = len(frequency_bands)
    # Setting up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot each frequency band
    ax.plot(rest_index/fs,average_AM_LPF_AC_premov[:,chi],color = 'g',linestyle='-',label='Premovement Stage')
    ax.plot(rest_index/fs,average_AM_LPF_AC_rest[:,chi],color = 'r',linestyle='-',label='Resting Stage')
    # Adding labels and title
    ax.set_xlabel('Lags[secs]', fontsize=12)
    ax.set_ylabel('AutoCorrelation', fontsize=12)
    ax.set_title(f'Autocorrealtion for {ch_names[chi]}', fontsize=14)
    # Adding a legend
    ax.legend(title='Movement Phases:')
    # Save the plot as a PDF file
    pdf_pages_AC.savefig(fig)
    # Closing the plot
    plt.close()
pdf_pages_AC.close()
del average_AM_LPF_AC_premov, average_AM_LPF_AC_rest, std_AM_LPF_AC_premov, std_AM_LPF_AC_rest
#%% AUTOCORRELATION after Average Mastoids and LPF
AM_LPF_all_AC_premov = np.stack([utils.compute_features(EEGEpochs_AM_LPF_all[i,premov_index,:], fs, 'autocorr') for i in range(EEGEpochs_AM_LPF_all.shape[0])])
AM_LPF_all_AC_rest = np.stack([utils.compute_features(EEGEpochs_AM_LPF_all[i,rest_index,:], fs, 'autocorr') for i in range(EEGEpochs_AM_LPF_all.shape[0])])

average_AM_LPF_AC_premov = np.mean(AM_LPF_all_AC_premov,axis = 0)
average_AM_LPF_AC_rest = np.mean(AM_LPF_all_AC_rest,axis = 0)

std_AM_LPF_AC_premov = np.std(AM_LPF_AC_premov,axis = 0)
std_AM_LPF_AC_rest = np.std(AM_LPF_AC_rest,axis = 0)

pdf_pages_AC = PdfPages(os.path.join(result_path,'overall','feature_visualization','averageAC_AMastoid_LPF.pdf'))
for chi in range(0,22):
    currentChannel = ch_names[chi]
    # Number of frequency bands
    n_bands = len(frequency_bands)
    # Setting up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot each frequency band
    ax.plot(rest_index/fs,average_AM_LPF_AC_premov[:,chi],color = 'g',linestyle='-',label='Premovement Stage')
    ax.plot(rest_index/fs,average_AM_LPF_AC_rest[:,chi],color = 'r',linestyle='-',label='Resting Stage')
    # Adding labels and title
    ax.set_xlabel('Lags[secs]', fontsize=12)
    ax.set_ylabel('AutoCorrelation', fontsize=12)
    ax.set_title(f'Autocorrealtion for {ch_names[chi]}', fontsize=14)
    # Adding a legend
    ax.legend(title='Movement Phases:')
    # Save the plot as a PDF file
    pdf_pages_AC.savefig(fig)
    # Closing the plot
    plt.close()
pdf_pages_AC.close()
del average_AM_LPF_AC_premov, average_AM_LPF_AC_rest, std_AM_LPF_AC_premov, std_AM_LPF_AC_rest

#%% FEATURE EXTRACTION
def extract_features(eeg_epoch,fs,ch_names,include_erp = True, include_PSD = True, include_AC = True, include_LPFERP = True):
    
    
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

#%% SVM TRAINING with different feature sets
evaluation_table = pd.DataFrame()
evaluation_table.index.name = 'Feature Combination'
evaluation_table.columns.name = 'Regularization Value'

feature_combinations = list(product([0, 1], repeat=4))
feature_combinations = feature_combinations[1:]
for feature_combo in feature_combinations:
    print(f'Feature Combo: {feature_combo}')
    train_features = []
    train_label =[]
    for epochi in range(EEGEpochs_AM_all.shape[0]):    
        rest_epoch = EEGEpochs_AM_all[epochi,rest_index,:]
        rest_features_epoch = extract_features(rest_epoch, fs, ch_names,include_erp=feature_combo[0],include_PSD=feature_combo[1],include_AC=feature_combo[2],include_LPFERP=feature_combo[3])
        train_features.append(rest_features_epoch)
        train_label.append(0)
        
        premov_epoch = EEGEpochs_AM_all[epochi,premov_index,:]
        premov_features_epoch = extract_features(premov_epoch, fs, ch_names,include_erp=feature_combo[0],include_PSD=feature_combo[1],include_AC=feature_combo[2],include_LPFERP=feature_combo[3])
        train_features.append(premov_features_epoch)
        train_label.append(1)
        
        # if (epochi+1)%100 ==0:
        #     print(f'Features are computed for {epochi+1}/{EEGEpochs_AM_all.shape[0]} trials.')
        #     print(f'Total number of rest features: {rest_features_epoch.shape[1]}')
        #     print(f'Total number of rest features: {premov_features_epoch.shape[1]}')
            
        del rest_epoch, premov_epoch
    train_features = np.vstack(train_features)
    
    print(f'Feature Extraction of Case {feature_combo} is done.')
    C_list = [0.5,0.1,0.05,0.01,0.005,0.001]
    for C_value in C_list:
        svm_clf  = make_pipeline(StandardScaler(),SVC(kernel='linear',C = C_value))
        scores = cross_val_score(svm_clf,train_features,train_label,cv = 5,scoring='accuracy')
        print(f'Accuracy Percentange (%): {round(scores.mean(),4)*100} +/- {round(scores.std(),4)*100} ')
        evaluation_table.loc[str(feature_combo),str(C_value)] = str(round(scores.mean()*100,4)) + '+/-' + str(round(scores.std()*100,4))
    print(f'Training of Case {feature_combo} is done.')
    
os.makedirs(os.path.join(result_path,'models')) if not os.path.exists(os.path.join(result_path,'models')) else None
evaluation_table.to_csv(os.path.join(result_path,'models','Linear_SVM.csv'))
        
        
#%% ENTROPY OF ERP after Average MASTOID
best_combo = [0,0,0,1]
train_features = []
train_label =[]
for epochi in range(EEGEpochs_AM_all.shape[0]):    
    rest_epoch = EEGEpochs_AM_all[epochi,rest_index,:]
    rest_features_epoch = extract_features(rest_epoch, fs, ch_names,include_erp=best_combo[0],include_PSD=best_combo[1],include_AC=best_combo[2],include_LPFERP=best_combo[3])
    train_features.append(rest_features_epoch)
    train_label.append(0)
    
    premov_epoch = EEGEpochs_AM_all[epochi,premov_index,:]
    premov_features_epoch = extract_features(premov_epoch, fs, ch_names,include_erp=best_combo[0],include_PSD=best_combo[1],include_AC=best_combo[2],include_LPFERP=best_combo[3])
    train_features.append(premov_features_epoch)
    train_label.append(1)
    
    if (epochi+1)%100 ==0:
        print(f'Features are computed for {epochi+1}/{EEGEpochs_AM_all.shape[0]} trials.')
        print(f'Total number of rest features: {rest_features_epoch.shape[1]}')
        print(f'Total number of rest features: {premov_features_epoch.shape[1]}')
        
    del rest_epoch, premov_epoch
train_features = np.vstack(train_features)
best_svm_clf  = make_pipeline(StandardScaler(),SVC(kernel='linear',C = 0.001))
best_svm_clf.fit(train_features,train_label)
import joblib
os.makedirs(os.path.join(result_path,'models')) if not os.path.exists(os.path.join(result_path,'models')) else None
# Save the pipeline
joblib.dump(best_svm_clf, os.path.join(result_path,'models','best_linear_SVM.pkl'))
#%% Storing it in .mat file
from scipy import io
feature_file_path = os.path.join(parent_dir,"data\\processed\\feature_.mat")
io.savemat(feature_file_path,{'time_axis': time_axis,'class_info': class_info_all[:,0].reshape(-1,1),'segmented_eeg': EEGEpochs_AM_all,'num_samples': class_info_all[:,1].shape[0],'num_channels':EEGEpochs_AM_all.shape[2],
                              'total_trials': EEGEpochs_AM_all.shape[1],'ch_names': ch_names})

#%% End of the script
sec1_exe_time = time.time()-start_timer_sec1
print(f'Data processing and feature extraction took {sec1_exe_time/60} minutes.')
    
    
    
    
    