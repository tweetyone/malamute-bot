import numpy as np
import pandas as pd


import os

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors

from scipy.spatial import distance

from gensim.models import KeyedVectors

import skipthoughts


class Match:

	def __init__(self):

		# Read data from file 'filename.csv'
		self.data = pd.read_csv("../../../data.csv",header=None)
		self.questions = self.data[1]

		self.answers = self.data[2]
		self.commonSts = ['Do you want to know more about UW?', 'What else do you want to know?','Is there any other information you would like to know?']

                model = skipthoughts.load_model()
                self.encoder = skipthoughts.Encoder(model)
                self.question_vectors = self.encoder.encode(self.questions)


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
		#s1_afv = self.avg_feature_vector(userquestion,model= self.embeddings_index, num_features=300)
                s1_afv = self.encoder.encode([userquestion])[0]
		min_cos = 10000
		idx = 0
		for i,q in enumerate(self.questions):
			#s2_afv = self.avg_feature_vector(q, model= self.embeddings_index, num_features=300)
                        s2_afv = self.question_vectors[i]
			cos = distance.cosine(s1_afv, s2_afv)
			if(cos < min_cos):
				min_cos = cos
				idx = i
		#print(idx, self.answers[idx])
		reply = self.answers[idx]+self.commonSts[np.random.randint(3)]
		return(reply)
