# -*- coding: utf-8 -*-
"""Q1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gjvNX7wlE8TTNAsZE08GRM4qmUWPEXco
"""

# import nltk
# from nltk.corpus import abc
# import numpy as np
# nltk.download('abc')
# nltk.download('punkt')

# punc = '''!()-[]{};:'°"+\,<>./?@#=$%^1234567890&*_~'''

# dic = np.unique(nltk.corpus.abc.words())
# for i in range(len(dic)):
#     dic[i] = dic[i].lower()
#     for j in dic[i]:
#         if j in punc:
#             dic[i] = dic[i].replace(j, "")
# dic = list(np.unique(list(filter(lambda x: x != "", dic))))

# sen = list(nltk.corpus.abc.sents())
# for i in range(len(sen)):
#     for k in range(len(sen[i])):
#         sen[i][k] = sen[i][k].lower()
#         for j in sen[i][k]:
#             if j in punc:
#                 sen[i][k] = sen[i][k].replace(j, "")
# for i in range(len(sen)):
#     sen[i] = list(filter(lambda x: x != "", sen[i]))

# #SUBSAMPLING
# allwordslist = []
# for i in sen:
#     for j in i:
#         allwordslist.append(j)
# from collections import Counter
# import random
# count = Counter(allwordslist)
# t = 0.00001
# tobedropped = []
# for i in dic:
#   p = 1 - np.sqrt(t/(count[i]/len(allwordslist)))
#   if(p > 1 - random.random()):
#     tobedropped.append(i)

# for i in range(len(dic)):
#     if dic[i] in tobedropped:
#       dic[i] = dic[i].replace(dic[i], "")
# dic = list(np.unique(list(filter(lambda x: x != "", dic))))
# for i in range(len(sen)):
#     for k in range(len(sen[i])):
#         if sen[i][k] in tobedropped:
#           sen[i][k] = sen[i][k].replace(sen[i][k], "")
# for i in range(len(sen)):
#     sen[i] = list(filter(lambda x: x != "", sen[i]))

# window_size = 2
# dim_hiddenlayer = 5

# def onehotencoding(arr, word):
#     ind = arr.index(word)
#     return ind
# def buildtrainingdata(sen, dic, window_size):
#     xtrain = []
#     ytrain = []
#     for i in sen:
#       if(len(i)>1):
#         for k in range(len(i)):
#             l = onehotencoding(dic, i[k])
#             for j in range(k-window_size, k+window_size):
#                 if j != k and j < len(i) and k >= 0:
#                   context = []
#                   xtrain.append(l)
#                   ytrain.append(onehotencoding(dic,i[j]))
#     return xtrain, ytrain
# xtrain,ytrain = buildtrainingdata(sen, dic, window_size)

from google.colab import drive
drive.mount('/mntDrive', force_remount=True)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pickle

# import pickle
# pickle.dump(dic,open('/mntDrive/My Drive/MCA_HW3/dic.pickle','wb'))
# pickle.dump(sen,open('/mntDrive/My Drive/MCA_HW3/sen.pickle','wb'))
# pickle.dump(xtrain,open('/mntDrive/My Drive/MCA_HW3/xtrain.pickle','wb'))
# pickle.dump(ytrain,open('/mntDrive/My Drive/MCA_HW3/ytrain.pickle','wb'))

dic=pickle.load(open('/mntDrive/My Drive/MCA_HW3/dic.pickle','rb'))
sen=pickle.load(open('/mntDrive/My Drive/MCA_HW3/sen.pickle','rb'))
xtrain=pickle.load(open('/mntDrive/My Drive/MCA_HW3/xtrain.pickle','rb'))
ytrain=pickle.load(open('/mntDrive/My Drive/MCA_HW3/ytrain.pickle','rb'))

np.shape(dic)

def createbatches(xtrain, ytrain, batchsize):
  nbatches = int(len(xtrain)/batchsize)
  for i in range(0,nbatches*batchsize,batchsize):
    yield xtrain[i:i+batchsize], np.array(ytrain[i:i+batchsize])[:,None]

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
x_train = tf.placeholder(tf.int32, shape = [None], name = 'x_train')
y_train = tf.placeholder(tf.int32, shape = [None,None], name = 'y_train')

hidden_layer_length = 10 
embedding = tf.Variable(tf.random_uniform([len(dic), hidden_layer_length], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embedding, x_train)



W = tf.Variable(tf.truncated_normal((len(dic), hidden_layer_length)))
b = tf.Variable(tf.zeros(len(dic)))
loss = tf.nn.sampled_softmax_loss(W, b, y_train, embed, 100, len(dic))
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer().minimize(cost)



words = 1000
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(100):
        loss = 0
        batches = createbatches(xtrain, ytrain, 1000)
        for i, j in batches:
            train_loss, _ = sess.run([cost, optimizer], feed_dict={x_train: i, y_train: j})  
            loss = loss + train_loss
        embed_mat = sess.run(embedding)
        embed_tsne = TSNE().fit_transform(embed_mat[:words, :])
        fig, ax = plt.subplots(figsize=(20, 20))
        for i in range(words):
            plt.scatter(*embed_tsne[i, :])
            plt.annotate(dic[i], (embed_tsne[i, 0], embed_tsne[i, 1]))
        print("Epoch =", ep+1, ", Average Training Loss =", loss/int(len(xtrain)/1000))
        plt.savefig("/mntDrive/My Drive/MCA_HW3/checkpoints/epoch{}.jpg".format(ep+1))
        plt.close()
    save_path = saver.save(sess, "/mntDrive/My Drive/MCA_HW3/checkpoints/mod.ckpt")

#References: https://towardsdatascience.com/word2vec-skip-gram-model-part-2-implementation-in-tf-7efdf6f58a27, https://github.com/mchablani/deep-learning/blob/master/embeddings/Skip-Gram_word2vec.ipynb

import glob
img = glob.glob('/mntDrive/My Drive/MCA_HW3/checkpoints/*.jpg')

import imageio
images = []
count = 0
for i in img:
  if count%10 == 0:
    images.append(imageio.imread(i))
  count+=1



imageio.mimsave('/mntDrive/My Drive/MCA_HW3/epochs10.gif', images)

































