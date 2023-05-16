# Distracted-Driver-Detection
A collection of models for distracted driver detection using PCA + KNN, Autoencoder + KNN, Custom CNNs, VGG16, RESNET, and MobileNET.

Aim : To detect if the car driver is driving safely or performing any activity that might result in an accident or any harm to others.

Dataset:
Dataset from Kaggle (State Farm Distracted Driver Detection).
Key Features of the State Farm Distracted Driver Detection Dataset are:
Number of images (Training set): 22,424.
Number of images (Testing set): 79,726.
Number of classes: 10
Image size: 640x480 pixels
Image format: JPG
Class distribution: uneven, ranging from 2,000 to 2,800 images per class
Data source and collection method: dashcams in actual vehicles driven by State Farm customers.

Methods:
With the understanding of the problem statement and several brainstorming sessions we moved on to creating models for our dataset. The basic idea was to construct the following:
1. PCA and Autoencoder with KNN classifier
2. Custom CNN 01
3. Custom CNN 02
4. VGG16
5. RESNET
6. MobileNET

Above models are initially trained over normal data splitting but, after we encountered the issue of data leaking. So, data is splitted over driver IDs to avoid appearance ot the same same driver in two or more splits. Afterwards trained above models again to encounter the data leaking issue.




