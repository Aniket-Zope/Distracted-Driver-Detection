Term Project - Image Processing with Machine Learning (DA 526)

## Distracted-Driver-Detection

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

Above models are initially trained over normal data splitting but, after encountered the issue of data leaking. So, data is splitted over driver IDs to avoid appearance of the same same driver in two or more splits. Afterwards trained above models again to encounter the data leaking issue.

Tools and Libraries Used:
1. Python: Programming language used for implementing the code.
2. PyTorch: Deep learning framework for building and training neural networks.  
3. TensorFlow: Deep learning framework for building and training neural networks. 
4. Keras: High-level neural networks API running on top of TensorFlow. 
5. NumPy: Library for numerical computations and array operations. 
6. Pandas: Library for data manipulation and analysis. 
7. Scikit-learn: Library for machine learning algorithms and evaluation metrics. 
8. OpenCV: Library for computer vision tasks, such as image preprocessing and manipulation. 
9. Matplotlib: Library for data visualization and plotting. 
10. Jupyter Notebook: Interactive environment used for code development and experimentation. 
11. Kaggle Notebook: Cloud-based notebook environment provided by Kaggle for running code and exploring datasets, etc. 




