# Enhancing Disease Detection with Machine and Deep Learning: Analyzing Chest X-rays for Pneumonia and Atelectasis Identification

### ABSTRACT:
Machine learning is a useful Artificial Intelligence (AI) tool that can help a lot in the medical field for obtaining an accurate prediction in test or laboratory results and give healthcare professionals a better understanding of the patients’ conditions before diagnosis. Deep Learning (DL) is another technique which makes use of advanced learning techniques and algorithms which can help to give a better output in certain cases. From using x-rays, CT, MRI, PET scan images, to using different signal and text datasets, an easier understanding of different diseases and their physical, mental and social conditions can be well understood using various ML and DL techniques and algorithms, such as k-Nearest Neighbors (kNN), Transfer Learning, Decision Tree methods like AdaBoost, Random Forest, Convolutional Neural Networks (CNN) like AlexNet and DenseNet, Artificial Neural Networks (ANN), etc. In this work, the area of interest lies in using chest x-ray (CXR) images, which depicts two major disease conditions – Pneumonia and Atelectasis. Atelectasis-infected patients would have smaller or disproportionate lungs due to the fall of the alveolar sac found in them. Hence, the lungs appear to be smaller than they are supposed to be. While in Pneumoniainfected patients, there are white patches found inside the pulmonary organs in the x-rays.

### DATASET DESCRIPTION:
The dataset used for this project comprises of three folders, each containing 200 Chest X-Ray images in jpg, jpeg and png formats. Each of the folders depict and consist of those x-rays related to three pulmonary conditions – Atelectasis, Pneumonia and Normal conditions. Out of the excess number of images available, 200 of them have been randomly taken and used for this project. The details regarding the datasets for each pulmonary condition are provided separately under references.

### METHODOLOGY:
A total of 6 models that have been used to detect pulmonary diseases – Decision Tree using AdaBoost algorithm, Random Forest, CNN using AlexNet and DenseNet architectures, Transfer Learning, and Support Vector Machines (SVM). Each has their own algorithms, Accuracy, Precision, Recall, F1-score, Support, Confusion Matrix, and Classification Report. The pre-processing of the images was done by simply converting all the x-ray images from colored to grayscale. This is done to identify the blackened pulmonary organs (atelectasis detection) and thesmall white patches (pneumonia detection). For all the models used, we have split each dataset into training and testing data in a 70:30 ratio.

#### A. CNN (DenseNet & AlexNet) -
Images are pre-processed by converting from grayscale to colored format and then resized. Both models are defined and compiled, with 10 epochs run. Sklearn matrices were used from python to print accuracy, confusion matrix, and classification report. Feature selection is as follows:

a) For AlexNet - Relevant features are selected and the images are classified accordingly.

b) For DenseNet - A global pooling 2D layer is added wherein it averages all the layers and combines it into a single feature vector, making it easier for classification.

#### B. Decision tree (AdaBoost) -
Target dimensions are set (here, we have selected 100 for both height and width) after pre-processing, and after training the model, required outputs using sklearn matrices were obtained. Sample weights found were initialised, with equal weights to each data point assigned. Then training using the Gini Index and calculating weighted error is done.

The amount of say is used to determine how much a classifier contributes to the final classifier in the first model. Hence, the higher the amount of say value, more influence it has on the final decision. Normalize these updated weights so that the sum of the weights never exceeds one. Further training is done, at which the weights keep
updating using the same steps as given above.

#### C. Random Forest -
Images are converted to colored (RGB) images and resized to specific target size and flattened into 1D arrays. The data then gets split into training and testing data, and the hyperparameters set were 100 trees to be made in the forest, and 42 for random seed number generation (for reproducibility).

#### D. Support Vector Machines (SVM) -
A Linear SVM is used, with normalizing the train and test data obtained. The regularization parameter is set to 1, which helps to improve accuracy, and a random state of 42 was used for reproducibility.

#### E. Transfer Learning - 
The base and top layers in the model are taken and compiled together and run. And after loading the pre-trained layers (by using DenseNet, without top layers), they are frozen. DenseNet121 using keras library was used. The sting labels which were used have been converted into integer labels which have been converted into hotencoded format, as is it used in multi-class classification. With 428 layers from DenseNet121, and 3 customized layers – Global Averaging Pool, a connecting layer with 1024 units and ReLu activation, and an outer layer with 3 units of classes and softmax classification.

### REFERENCES:
#### PAPERS/ARTICLES:
1. M. Ayalew, Y. A. Bezabiah, B. M. Abuhayi, and A. Ayalew, “Atelectasis detection in chest X-ray images using convolutional neural networks and transger learning with anisotropic diffusion filter,” Informatics in Medicine Unlocked, vol. 45, p. 101448, Jan. 2024.
2. E. J. R. Van Beek, J. S. Ahn, M. J. Kim, and J. Murchison, “Validation study of machine-learning chest radiograph software in primary and emergency medicine,” Clinical Radiology, vol. 78, no. 1, pp. 1–7, Jan. 2023.
3. X. Huang et al., “External validation based on transfer learning for diagnosing atelectasis using portable chest X-rays,” Frontiers in Medicine (Lausanne), vol. 9, Jul. 2022.
4. S. Goyal and R. Singh, “Detection and classification of lung diseases for pneumonia and Covid-19 using machine and deep learning techniques,” Journal of Ambient Intelligence and Humanized Computing, Sep. 2021.
5. T. B. Chandra and K. Verma, “Pneumonia detection on chest X-Ray using machine learning paradigm,” in Advances in intelligent systems and computing (Internet), 2019, pp. 21–33.
6. T. Rahman et al., “Transfer Learning with Deep Convolutional Neural Network (CNN) for Pneumonia Detection Using Chest X-ray,” Applied Sciences (Basel), vol. 10, no. 9, p. 3233, May 2020.
7. “A Review on Detection of Pneumonia in Chest X-ray Images Using Neural Networks,” Journal of Biomedical Physics and Engineering,
vol. 12, no. 6, Dec. 2022.
8. A. A. Nasser and M. A. Akhloufi, “Deep learning methods for chest disease detection using radiography images,” SN Computer Science, vol. 4, no. 4, May 2023.
9. S. Sharma and K. Guleria, “A Deep Learning based model for the Detection of Pneumonia from Chest X-Ray Images using VGG-16 and Neural Networks,” Procedia Computer Science, vol. 218, pp. 357–366, Jan. 2023.
10. R. Kundu, R. Das, Z. W. Geem, G.-T. Han, and R. Sarkar, “Pneumonia detection in chest X-ray images using an ensemble of deep learning models,” PLoS ONE, vol. 16, no. 9, p. e0256630, Sep. 2021.
11. A. K. Jaiswal, P. Tiwari, S. Kumar, D. Gupta, A. Khanna, and J. J. P. C. Rodrigues, “Identifying pneumonia in chest X-rays: A deep learning approach,” Measurement, vol. 145, pp. 511–518, Oct. 2019.
12. P. Raghav, “CNNArchitectures, - LetNet, AlexNet, VGG, GoogleNet and ResNet,” Medium, Mar. 15, 2018. Accessed: Apr. 15, 2024. Available: https://medium.com/@RaghavPrabhu/cnn-architectureslenet-alexnet-vgg-googlenet-and-resnet-7c81co17b848
13. P. Ruiz, “Undertanding and Visualizing DenseNets,” Medium, Oct. 18 2018. Accessed: Apr. 15, 2024. Available: https://towardsdatascience.com/understanding-and-visualizingdensenets-7f688092391a
14. “Convolutional Neural Network (CNN) Architecture,” GeeksforGeeks, Mar. 21, 2023. Accessed: Apr. 15, 2024. Available: https://www.geeksforgeeks.org/convoluyional-neural-network-cnnarchitectures/
15. “Adaboost Algorithm Explained in Depth,” ProjectPro, Mar. 20 2024. Accessed: Apr. 14 2024. Available: https://www.projectpro.io/article/adaboost-algorithm/972
16. A. Kalbande, “Random Forest Algorithm in Machine Learning,” Fireblaze AI School, Sep. 25, 2020. Accessed: Apr. 14, 2024. Available: https://www.fireblazeaischool.in/blogs/random-forestalgorithm/

#### DATA:
**For Atelectasis Dataset -** K. Mryou, “Atelectasis,” Kaggle, Aug. 11, 2021. Accessed: Feb. 4, 2024. Available: https://www.kaggle.com/datasets/kheyduzoomryou/pneumothorax

**For Normal Dataset -** P. Mooney, “Chest X-Ray images (Pneumonia),” Kaggle, Mar. 24, 2018. Accessed: Feb. 4, 2024. Available: https://www.kaggle.com/datasets/paulimothymooney/chest-xraypneumonia

**For Pneumonia Dataset -** P. Mooney, “Chest X-Ray images (Pneumonia),” Kaggle, Mar. 24, 2018. Accessed: Feb. 4, 2024. Available: https://www.kaggle.com/datasets/paulimothymooney/chest-xraypneumonia
