import numpy as np 
import pandas as pd 
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pylab import rcParams

def read_data(file_name):
    with open(file_name,'r') as f:
        word_vocab = set() 
        word2vector = {}
        for line in f:
            line_ = line.strip() 
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:],dtype=float)
    print("Total Words in DataSet:",len(word_vocab))
    return word_vocab,word2vector

vocab, w2v = read_data("glove.6B.50d.txt")

def find_w4(word1,word2,word3, w2v):
    word_list = w2v.keys()
    max_sim = -1000
    word1,word2,word3 = word1.lower(),word2.lower(),word3.lower()
    diff_vec = w2v[word3] + (w2v[word1]-w2v[word2]) 
    for word in word_list:
        vec = w2v[word]
        sim_ = cos_sim(u=diff_vec,v=vec)
        if sim_ > max_sim:
            max_sim = sim_
            word_selected =  word
            
    return word_selected

def cos_sim(u,v):

    numerator_ = u.dot(v)
    denominator_= np.sqrt(np.sum(np.square(u))) * np.sqrt(np.sum(np.square(v)))
    return numerator_/denominator_

all_words = w2v.keys()
print("Similarity Score of King and Queen",cos_sim(w2v['king'],w2v['queen']))

def return_matrix(random_words,dim =50):
    word_matrix = np.random.randn(len(random_words),dim)
    i = 0
    for word in random_words:
        word_matrix[i] = w2v[word]
        i +=1
    return word_matrix

print("King - man + woman = ",find_w4('king','man','woman',w2v))

