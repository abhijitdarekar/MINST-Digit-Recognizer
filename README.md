# MINST Digit Recognizer

AIM : Train a pytorch model on MINST Dataset to for handwritten digit recognition and deploy it in AZURE MLOps servies.

Pytorch model is trained on MISNT dataset, and an api application is built to recieve input image and respond with number recognised by model.

app.py : Fast api server for api management.
/src  : containes files related to model training and for realtime prediction.
/notebooks : Contains notebooks realated to model training and testing
/arifacts : All the reusbale data files/ model pickle files in placed here
