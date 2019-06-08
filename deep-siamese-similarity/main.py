# -*- coding: utf-8 -*-
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import sys
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper
import pandas as pd
from gensim.models import KeyedVectors


# embedding_path = 'GoogleNews-vectors-negative300.bin'
# embedding_dim = 300
# max_seq_length = 10
# savepath = './data/en_SiameseLSTM.h5'
# embedding_dict = KeyedVectors.load_word2vec_format(embedding_path, binary=True)

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 21, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filepath", "miniQA.txt", "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("vocab_filepath", "runs/1559975068/checkpoints/vocab", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "runs/1559975068/checkpoints/model-9000", "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS(sys.argv)

database = pd.read_csv('questions_answers.csv')
questions = database['Question'].values
answers = database['Answer'].values

inpH = InputHelper()
x1_test,x2_test,y_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, 30)

checkpoint_file = FLAGS.model
print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/distance").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

        batches = inpH.batch_iter(list(zip(x1_test,x2_test,y_test)), FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_d=[]
        for db in batches:
            x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
            batch_predictions, batch_acc, batch_sim = sess.run([predictions,accuracy,sim], {input_x1: x1_dev_b, input_x2: x2_dev_b, input_y:y_dev_b, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            print(batch_predictions)
            all_d = np.concatenate([all_d, batch_sim])
            print("DEV acc {}".format(batch_acc))
        for ex in all_predictions:
            print ex
        correct_predictions = float(np.mean(all_d == y_test))
        print("Accuracy: {:g}".format(correct_predictions))

#
# if __name__ == '__main__':
#
#     while True:
#         max_score = 0
#         correctQ = ""
#         sen1 = input("input sentence1: ")
#         for q in questions:
#             sen2 = q
#             print(sen2)
#             dataframe = pd.DataFrame({'question1': ["".join(sen1)], 'question2': ["".join(sen2)]})
#
#             dataframe.to_csv("./data/test.csv", index=False, sep=',', encoding='utf-8')
#             TEST_CSV = './data/test.csv'
#
#             # 读取并加载测试集
#             test_df = pd.read_csv(TEST_CSV)
#             for q in ['question1', 'question2']:
#                 test_df[q + '_n'] = test_df[q]
#                 #print(test_df)
#
#             # 将测试集词向量化
#             test_df, embeddings = make_w2v_embeddings(flag, embedding_dict, test_df, embedding_dim=embedding_dim)
#
#             # 预处理
#             X_test = split_and_zero_padding(test_df, max_seq_length)
#
#
#             # 确认数据准备完毕且正确
#             assert X_test['left'].shape == X_test['right'].shape
#
#             # 预测并评估准确率
#             prediction = model.predict([X_test['left'], X_test['right']])
#             print(prediction)
#             if (prediction >= max_score):
#                 max_score = prediction
#                 correctQ = sen2
#
#         print(correctQ,max_score)
