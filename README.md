# NLP_Classifier
Classifying text based on various ML approaches 

Trying out various NLP approaches to classify text.

#1. classifier_naive_bayes.py  
Usage:  
~ python classifier_naive_bayes.py input_file    
The input files should contain comma saperated labled sentences which will be used to train the model. ( test1.csv & test2.csv are samples)
The program will remove stop words, convert the text to lower case and then train a model using multinomail Naive Baiyes classifier & n-gram appraoch. 
Accuracy of the model improves significantly when the size of training data is increased. 

#2. classifier_tensor.py  
Usage:    
~ python classifier_tensor.py input_file  
WIP  
Trying to use tensorflow algorithms to solve the classification problem. Using tokenizer package along with padding to tokenzinze and pad the sentences. 
Trying to create a neural network which has an embedding layer and pooling layer . 
