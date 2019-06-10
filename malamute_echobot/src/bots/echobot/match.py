import numpy as np
import pandas as pd


import os

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors

from scipy.spatial import distance

from gensim.models import KeyedVectors


class Match:

	def __init__(self):

		# Read data from file 'filename.csv'
		self.data = pd.read_csv("/Users/gaoxingyun/Documents/uw/courses/Sp19/EE596_ConvAI/ee596_spr2019_lab1/data.csv",header=None)
		self.questions = self.data[1]

		self.answers = self.data[2]
		self.commonSts = ['Do you want to know more about UW?', 'What else do you want to know?','Is there any other information you would like to know?']


		self.filepath = "/Users/gaoxingyun/Documents/uw/courses/Sp19/EE596_ConvAI/ee596_spr2019_lab1/GoogleNews-vectors-negative300.bin"


		self.wv_from_bin = KeyedVectors.load_word2vec_format(self.filepath, binary=True)
		#extracting words7 vectors from google news vector
		self.embeddings_index = {}
		for word, vector in zip(self.wv_from_bin.vocab, self.wv_from_bin.vectors):
		    self.coefs = np.asarray(vector, dtype='float32')
		    self.embeddings_index[word] = self.coefs


	def avg_feature_vector(self,sentence, model, num_features):
		words = sentence.split()
		#feature vector is initialized as an empty array
		feature_vec = np.zeros((num_features, ), dtype='float32')
		n_words = 0
		for word in words:
			if word in self.embeddings_index.keys():
				n_words += 1
				feature_vec = np.add(feature_vec, model[word])
		if (n_words > 0):
			feature_vec = np.divide(feature_vec, n_words)
		return feature_vec

	def bestMatch(self,userquestion):
		s1_afv = self.avg_feature_vector(userquestion,model= self.embeddings_index, num_features=300)
		min_cos = 10000
		idx = 0
		for i,q in enumerate(self.questions):
			s2_afv = self.avg_feature_vector(q, model= self.embeddings_index, num_features=300)
			cos = distance.cosine(s1_afv, s2_afv)
			if(cos < min_cos):
				min_cos = cos
				idx = i
		#print(idx, self.answers[idx])
		reply = self.answers[idx]+self.commonSts[np.random.randint(3)]
		return(reply)
