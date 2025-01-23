# Description of Folders:
\app: stores the files required for application development including the icon. 
\archive: stores scripts or other files that are old. 
\data: stores raw and processed data that is required for the project. 
\docs: stores the documents related to the project. I have stores all the updates and presentation documents in this folder. 
\results: stores the results of the analysis as well as the training of classification model. 
\src: stores the Python scripts that was coded for this project.

# HOW TO RUN THE PROJECT 
Three main codes:
- \src\utils.py: This script defines all the required utility functions that are used in this project. 
- \src\feature_engineering_v01.py: This script provides the code for EEG extraction, feature analysis, training the model to find the best model, and training the best mode and saving the model to be reused in \src\implementation_GUI.py.
- \src\implementation_GUI.py: it used the trained model to predict the new EEG dataset that is extracted using the features of the GUI.