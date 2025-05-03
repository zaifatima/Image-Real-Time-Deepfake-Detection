-Deepfake Detection Using Deeplearning Algorithms 
-Real time and Image detection 
-Resnet-18 is used for Realtime Camera detection and CNN for Realtime Deepfake Detection
-The CNN model leverages an accuracy of 83% while Resnet-18 of 98%
-CNN was trained on ~1400000 and Resnet-18 on ~40000 images.
Working:
Checkpoints folder containes Cnn weights and Architecture separately file names being (cnn_epoch_25.weights.h5 and cnn_architecture.json respectively) train_model was used to train cnn.
best_model.pth is combined weights and architecure of the resnet-18 model, RN18.py was used to train it.
test.ipnyb shows both model's accuracy, confusion matrices and classification matrices.
app.py is the file that combines all the features and facilitates the working of the models in a flask app.
Templates folder containes 4 html files that deals with 4 different interfaces.
requirements.txt contains all of the necessary requirements.
Python version used-3.11.5
Dataset- Real and Deepfake Images (~2GB) from Kaggle.
Note: Dataset isn't uploaded due to size constrains as well as additional private images added into the dataset for training to leverage a better performance with real world data. 
