# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 00:10:01 2024

@author: brth229
"""

import sys
import numpy as np
import joblib
import scipy
from scipy import signal
from utils import utils
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
#%% PyQT Based Libraries
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QHBoxLayout

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtGui import QFont

#%% Clearing the variables and initilizing the timer 
from IPython import get_ipython;   
try:
    get_ipython().run_line_magic('reset', '-sf')
except AttributeError:
    print("Not running in an IPython environment. Skipping reset.")


import time
script_start = time.time()
import os
# Changing the current directory to the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir));
app_dir = os.path.join(parent_dir, "app")

#%% LOADING SVM
# Load the trained pipeline
loaded_pipeline = joblib.load(os.path.join(parent_dir,'results','models','best_linear_SVM.pkl'))



# Assuming test_features are defined
# test_features.shape -> (m, n_features)

# Use the loaded pipeline to predict new data
# test_predictions = loaded_pipeline.predict(test_features)

# # Evaluate accuracy if test labels are available
# from sklearn.metrics import accuracy_score

# # Assuming test_labels are defined
# test_accuracy = accuracy_score(test_labels, test_predictions)
# print("Test Accuracy:", test_accuracy)


#%% 

#%% Import your EEG processing and prediction modules

class EEGVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Self-paced Pre-movement Intention Detector')
        self.setGeometry(100, 100, 1000, 800)
        
        #Set an application icon
        self.setWindowIcon(QIcon(os.path.join(app_dir, 'ICON.png')))
        QApplication.setWindowIcon(QIcon(os.path.join(app_dir, 'ICON.png')))
        
        # Main layout
        layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        #Create a horinzontal layout for the bottons
        button_layout = QHBoxLayout()

        # Load EEG Button
        self.loadButton = QPushButton('Load EEG Signal')
        self.loadButton.clicked.connect(self.load_eeg)
        button_layout.addWidget(self.loadButton)
        
        # Add Pause and Continue buttons
        self.pauseButton = QPushButton('Pause')
        self.pauseButton.clicked.connect(self.pause_stream)
        button_layout.addWidget(self.pauseButton)

        self.continueButton = QPushButton('Continue')
        self.continueButton.clicked.connect(self.continue_stream)
        button_layout.addWidget(self.continueButton)
        
        #Add the button layout to the main layout
        layout.addLayout(button_layout)
        
        # Prediction Widhet
        self.predictionWidget = PredictionWidget(self)
        layout.addWidget(self.predictionWidget)
        

        
        # EEG Plot
        self.eegPlot = EEGPlot(self, width=8, height=4)
        layout.addWidget(self.eegPlot)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)

        #Diplay the default for both the plots
        default_data = np.zeros((150,19))
        ch_names = ['FP1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','A1','A2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz','X5']
        bad_indices = [ch_names.index('A1'),ch_names.index('A2'),ch_names.index('X5')]
        valid_channel_indices = [i for i in range(0, len(ch_names)-1) if i not in bad_indices]
        # Use valid_channel_indices to get the labels of valid channels
        channel_labels = [ch_names[i] for i in valid_channel_indices]
        self.eegPlot.plot(default_data,channel_labels,0,150)
        
        
        
        # Default style for buttons
        self.default_style = "QPushButton {}"
        self.active_style = "QPushButton { background-color: green; color: white; }"

        # Exit Button
        self.exitButton = QPushButton('Exit')
        self.exitButton.clicked.connect(self.close)  # Connect the button's clicked signal to the close slot
        button_layout.addWidget(self.exitButton)

    def load_eeg(self):
        default_data_path = os.path.join(parent_dir,'data','raw')
        filePath, _ = QFileDialog.getOpenFileName(self, "Load EEG File",default_data_path)
        if filePath:
            self.data_struct = scipy.io.loadmat(filePath)['o']
            self.eeg_data_raw = self.data_struct['data'].item()
            self.ch_names = [item[0][0] for item in self.data_struct['chnames'].item()]
            self.bad_indices = [self.ch_names.index('A1'),self.ch_names.index('A2'),self.ch_names.index('X5')]
            self.ear_channel_indices = [self.ch_names.index('A1'),self.ch_names.index('A2')]
            self.valid_channel_indices = [i for i in range(0, len(self.ch_names)-1) if i not in self.bad_indices]
            self.eeg_data = self.eeg_data_raw[:,]
            self.sample_rate = self.data_struct['sampFreq'][0][0].item()
            self.raw_markers = self.data_struct['marker'].item()
            self.modified_markers = np.where((self.raw_markers==1) | (self.raw_markers==2),1,0)
            
            print(f'Length of the EEG signal is {self.eeg_data.shape[0]/self.sample_rate*60} minutes.')
            self.data_stream = EEGDataStream(self.eeg_data,self.modified_markers)
            self.refresh_time = 250
            self.timer.start(self.refresh_time)  # Refresh every 100 ms
            
            print('File loaded successfully.')
        else:
            print("No file selected")
    def update_plot(self):
        if hasattr(self, 'data_stream'):
            try:
                data_segment, marker_segment,window_start_index,window_end_index = next(self.data_stream.get_next_window())
                ch_names = ['FP1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','A1','A2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz','X5']
                bad_indices = [ch_names.index('A1'),ch_names.index('A2'),ch_names.index('X5')]
                valid_channel_indices = [i for i in range(0, len(ch_names)-1) if i not in bad_indices]
                # Use valid_channel_indices to get the labels of valid channels
                channel_labels = [ch_names[i] for i in valid_channel_indices]
                
                
                self.eegPlot.plot(data_segment[:,valid_channel_indices],channel_labels,window_start_index,window_end_index)
                true_label = marker_segment[-1]==1
                
                ## Predicting using the model
                best_combo = [0,0,0,1]
                test_features = utils.extract_features(data_segment, 200, ch_names,include_erp=best_combo[0],include_PSD=best_combo[1],include_AC=best_combo[2],include_LPFERP=best_combo[3])
                test_prediction = loaded_pipeline.predict(test_features)
                #####################################
                
                pred_label = test_prediction
                self.predictionWidget.updatePrediction(true_label,pred_label,window_start_index,window_end_index)
                
            except StopIteration:
                self.timer.stop()  # No more data to read, stop the timer
                print('End of the detection for the data file.')
        else:
            print('Data is not loaded.')
            
    def pause_stream(self):
        self.timer.stop()
        self.pauseButton.setStyleSheet(self.active_style)
        self.continueButton.setStyleSheet(self.default_style)
        print("Detection paused.")
    def continue_stream(self):
        if hasattr(self, 'data_stream'):
            self.timer.start(self.refresh_time)
            self.pauseButton.setStyleSheet(self.default_style)
            self.continueButton.setStyleSheet(self.default_style)
            print("Detection continued.")
        else:
            print("No data to detect.")

class EEGPlot(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(EEGPlot, self).__init__(fig)
        self.setParent(parent)

    def plot(self, data,channel_labels,start_index,end_index):
        self.axes.clear()
        offset = 15
        channel_offsets = np.arange(0, data.shape[1] * offset, 15)
        for i in range(data.shape[1]):  # Loop through all channels
            self.axes.plot(np.arange(start_index, end_index)/200, data[:,i] + offset*i, label=f'Channel {i+1}')  # Offset each channel
        self.axes.set_title('EEG Signal')
        self.axes.set_xlabel('Time[sec]')
        self.axes.yaxis.set_ticks_position('none')
        self.axes.set_yticks(channel_offsets)
        self.axes.set_yticklabels(channel_labels)
        self.draw()

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class PredictionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)  # Main layout for the widget
        
        # First Matplotlib figure and axis for True vs Predicted
        self.truevspred = plt.figure(figsize=(8, 2))
        self.canvas1 = FigureCanvas(self.truevspred)
        self.main_layout.addWidget(self.canvas1)

        # Second box for info
        self.info_box = QLabel(self)
        self.info_box.setText("Information Box")
        self.info_box.setAlignment(QtCore.Qt.AlignCenter)
        self.info_box.setStyleSheet("background-color: lightgray; font-size: 16px;")
        self.main_layout.addWidget(self.info_box)

        # Initialize data storage for plotting
        self.time_data = []
        self.true_data = []
        self.pred_data = []
        self.additional_data = []  # For the second graph

    def updatePrediction(self, true, pred, start_time, end_time):
        # Append new data
        time_point = (start_time + end_time) / 2 / 200  # Center of the time window
        self.time_data.append(time_point)
        self.true_data.append(int(true))
        self.pred_data.append(int(pred))

        # Clear and plot the first graph (True vs Predicted Labels)
        self.figure1.clear()
        ax1 = self.figure1.add_subplot(111)
        ax1.plot(self.time_data, self.true_data, 'g-', label='True', linewidth=2)
        ax1.plot(self.time_data, self.pred_data, 'b-', label='Prediction', linewidth=2)
        
        ax1.set_title('True vs. Predicted Labels')
        ax1.set_xlabel('Time [sec]')
        ax1.set_ylabel('Label')
        ax1.set_ylim(-0.1, 1.1)  # Binary labels 0 and 1
        ax1.legend(loc='upper right')
        
        # Update the info box with dynamic content
        true_label_str = "True Label: " + ("Positive" if true else "Negative")
        pred_label_str = "Predicted Label: " + ("Positive" if pred else "Negative")
        self.info_box.setText(f"{true_label_str}\n{pred_label_str}\nTime Window: {start_time/200:.2f} to {end_time/200:.2f} sec")
        
        # Redraw the canvase
        self.canvas1.draw()


class EEGDataStream:
    def __init__(self, eeg_data,marker_data, sample_rate=200):
        # Load the entire data file at once
        self.eeg_data = eeg_data
        self.sample_rate = sample_rate
        self.raw_markers = marker_data
        self.window_size = int(0.75 * sample_rate)+1  # Number of samples in 0.75 seconds
        self.shift_size = int(0.25 * sample_rate)   # Number of samples to shift the window
        self.current_position = int(85*sample_rate)

    def get_next_window(self):
        """ Yield the next window of data until the end of the file """
        while self.current_position + self.window_size <= len(self.eeg_data):
            start_index = self.current_position
            end_index = start_index + self.window_size
            self.current_position += self.shift_size
            window_data = self.eeg_data[start_index:end_index,:]
            window_markers = self.raw_markers[start_index:end_index,0]
            # self.current_position += self.shift_size
            yield (window_data,window_markers,start_index,end_index)
    def get_current_position(self):
        return self.current_position
#%% MAIN FUNCTION
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EEGVisualizer()
    ex.show()
    sys.exit(app.exec_())
