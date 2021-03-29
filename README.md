This repository has 7 folders corresponding to 7 different tasks involving Deep Learning in various applications

A few of the tasks were done as part of 2IMM10-Recommender Systems or Deep Learning coursework

1. Anomaly_Detection_TS: (anomaly detection in timeseries data)

Anomalies are common in any timeseries data. But certain malfunctions that create anomalies or anomalies that creates impact are inevitable, but they can be predicted or analyzed to extract more information. This concept can find its application right from finance sector and health sector to sapce monitoring.

This task explains how to build an anomaly detector using LSTM autoencoders. Welcome to this project!


2. Co-occurrence matrix, CBOW and Skipgram models: 

The alice in wonderland text is read my the model, co-occurrence matrix is created to map the words in multi-dimensional space. This matrix is then used to find the similarity between the words and to find which set of words always come together. However, this multi-dimsional space is computationally costly, so other efficient methods such as CBOW and Skipgram models are built.

Word-embeddings are formed for CBOW and Skipgram models, thus reducing the computational complexity. The advantages of negative-sampling is also evidently observed while training and tesing. The main drawback of these models is that they cannot preserve the order of occurrence of words. (Sounds interesting? open the corresponding .ipynb file, test it yourself and feel free to drop in your comments/obervation)


3. Image Caption Generation

If an image is given to the model, can it generate a caption for it? Yes, it can be done by extracting the features in the image using Dense layers, RNN with LSTM/GRU to generate the caption in a meaningfull way. In addition, Greedy Search and Beam search algorithms are used for the same and their performances are compared.


4. Sentiment Analysis - Document Level and Aspect Level

This file performs the document level sentiment analysis using both the Uni-directional and Bi-directional RNNs and their performances are compared. Similarly, the model for aspect level sentiment classification is trained and tested to evaluate its performance


5. Siamese network, triplet network and one-shot-learining: 

In the standard image classification task using neural network, a dataset with many examples for each class of images is formed, a large amount of time is spent in training/re-training the NN model to get good accuracy. However, there are many situations in which there are a very few examples for certain class of images. In such a situation how can the NN model be trained? how do we split the training and testing set to get significant performance? and how can we solve other challenges that depends on large sample size for training?

After training the NN model on only one sample of a particular class of image, can this model classify any image from that particular class with good accuracy instead of random guessing? Yes, the NN model can be trained to do that using One-shot learning. 

Here Cifar-100 dataset is used for training the built siamese network. In the dataset, only 80 classes are used for training and the remaining 20 classes are used for one-shot testing. The one-shot learning is done by extracting the neural codes of the siamese network.

The same task is again repeated with Triplet network and the performances are compared.


6. TransferLearning-Fine_Tune_BERT_for_Text_Classification_with_TensorFlow

This assignment involves transfer learning, transformers network in NLP application using the tensorflow-hub's pretrained BERT model. The text can be translated to another language using RNN. In another application, sentiment analysis can be done only after receiving the full text, also it can be done done only to certain extent. However, by using Transformers network, the speed at which the texts can be processed can be increased as the sequential order of analyzing the text is removed and parallel computing is done using transformer network. One of the NLP applications is to classify if the given text is sincere or insincere using Quora database. Open the folder to gain more knowledge on this and the materials to read the required concepts for this project.


7. Variational Auto Encoder and Semi Supervised Learning

This file has two tasks. First is VAE, in this the encoder and decoders are designed, reparameterized, used KL loss, trained and tested using MNIST dataset. The encoder gives 2 values in its bottle-neck layer, the decoder picks this up and reconstructs the image giving it translational/pahse shift.

Second is the semi-supervised learning. This is a mixture of unsupervised and supervised learning. Not all the images in the real-world come with lables, this method is used to label the images that are present and then training the supervised learning model would give the desired result, thus reducing the labelling time. First find the similarity in features in the given images, so similar images will be clustered and a label would be given. 

Here, DAE is used, the guassian white noise is added to the image and the model is trained to identify and remove the noise. A good performance is achieved. Then, new images using the designed VAE network is generated. It is interesting to know that, the model has generated a hoodie which is not present in the dataset. Thus, this confirms that the VAE network is trained correctly to generate synthetic data. Then semi-suervised learning is performed and the performance is found to be good.





