# ADD THE LIBRARIES YOU'LL NEED
import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

'''
About the task:

You are provided with a codeflow- which consists of functions to be implemented(MANDATORY).

You need to implement each of the functions mentioned below, you may add your own function parameters if needed(not to main).
Execute your code using the provided auto.py script(NO EDITS PERMITTED) as your code will be evaluated using an auto-grader.
'''


def encode_data(text):
    # This function will be used to encode the reviews using a dictionary(created using corpus vocabulary) 
    
    # Example of encoding :"The food was fabulous but pricey" has a vocabulary of 4 words, each one has to be mapped to an integer like: 
    # {'The':1,'food':2,'was':3 'fabulous':4 'but':5 'pricey':6} this vocabulary has to be created for the entire corpus and then be used to 
    # encode the words into integers 

    # return encoded examples
    vocab = []
    for tokens in text:
        vocab = vocab+tokens
    vocab = set(vocab)
    
    encoder = dict()
    count = 1
    for word in vocab:
        encoder[word] = count
        count = count +1
    
    text_encoded = []
    
    for tokens in text:
        vec = []
        for word in tokens:
            vec.append(encoder[word])
        text_encoded.append(vec)
    
    return text_encoded


def convert_to_lower(text):
    # return the reviews after convering then to lowercase
    temp = []
    for sentence in text:
        s = sentence.lower()
        temp.append(s)
    return temp


def remove_punctuation(text):
    # return the reviews after removing punctuations
    temp = []
    for sentence in text:
        res = re.sub(r'[^\w\s]', '', text)
        temp.append(res)
    return temp

def remove_stopwords(text):
    # return the reviews after removing the stopwords
    temp = []
    stop_words = set(stopwords.words('english'))
    for sentence in text:
        tokens =  word_tokenize(sentence)
        new_sent = []
        for w in tokens:  
            if w not in stop_words:  
                new_sent.append(w)
        new_sent = str(new_sent)
        temp.append(new_sent)
    return temp

def perform_tokenization(text):
    # return the reviews after performing tokenization
    temp = []
    for sentence in text:
        tokens =  word_tokenize(sentence)
        temp.append(tokens)
    return temp

def perform_padding(data):
    # return the reviews after padding the reviews to maximum length
    max_l = 0
    for vec in data:
        if max_l<len(vec):
            max_l = len(vec)
    appended_data = data
    for i,vec in enumerate(appended_data):
        if len(vec)<max_l:
            appended_data[i] = vec + list(np.zeros(max_l-len(vec)))
    return appended_data

def preprocess_data(data):
    # make all the following function calls on your data
    # EXAMPLE:->
        '''
        review = data["reviews"]
        review = convert_to_lower(review)
        review = remove_punctuation(review)
        review = remove_stopwords(review)
        review = perform_tokenization(review)
        review = encode_data(review)
        review = perform_padding(review)
        '''
    # return processed data
    review = data["reviews"]
    review = convert_to_lower(review)
    review = remove_punctuation(review)
    review = remove_stopwords(review)
    review = perform_tokenization(review)
    review = encode_data(review)
    review = perform_padding(review)
    return review



def softmax_activation(x):
    # write your own implementation from scratch and return softmax values(using predefined softmax is prohibited)
    num = np.exp(x)
    den = np.sum(num)
    return num/den


class NeuralNet:

    def __init__(self, reviews, ratings):

        self.reviews = reviews
        self.ratings = ratings



    def build_nn(self):
        #add the input and output layer here; you can use either tensorflow or pytorch
        

    def train_nn(self,batch_size,epochs):
        # write the training loop here; you can use either tensorflow or pytorch
        # print validation accuracy

    def predict(self, reviews):
        # return a list containing all the ratings predicted by the trained model



# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):
    
    batch_size,epochs=
    
    train_reviews=preprocess_data(train_data)
    test_reviews=preprocess_data(test_data)

    model=NeuralNet(train_reviews,train_ratings)
    model.build_nn()
    model.train_nn(batch_size,epochs)

    return model.predict(test_reviews)