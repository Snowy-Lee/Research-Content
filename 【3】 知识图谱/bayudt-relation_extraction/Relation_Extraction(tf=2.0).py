#!/usr/bin/env python
# coding: utf-8

# #### Load Data

# In[91]:


import sys
import os
import pandas as pd
import numpy as np
import _pickle as cPickle
from numpy import array
import operator
import random
from fuzzywuzzy import fuzz

import tensorflow as tf
# 启用动态图机制
#tf.enable_eager_execution()    #2.0不需要此举
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import time
#from tensorflow.python.client import device_lib   #新增


INPUT_LENGTH = 32
OUTPUT_LENGTH = 14
TOTAL_EPOCHS = 20
BEAMSEARCH = True
POSPROCESSING = True

##################################新增的代码################################

DEVICE = "0"     #0是GPU，1不存在GPU，用cpu
os.environ["CUDA_VISIBLE_DEVICES"]=DEVICE    #运行设备
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# gpus = tf.config.list_physical_devices("GPU")
# print('GPU use:', gpus)
# print('GPU used:', tf.test.is_gpu_available())
# print('cuda used:',tf.test.is_built_with_cuda())


# In[92]:


#target_embedding和source_embedding的区别和作用？又是从何而来。不需要的之从何而来也能直接使用，但是使用的领域未知
#word_embedding是所有单词的embedding，而target_embedding是只有实体的embedding
word_to_idx = cPickle.load(open("data/ready/encoder-decoder-model-training/source_embedding_dict.pickle", "rb"))
idx_to_word = {v: k for k, v in word_to_idx.items()}

ent_to_idx = cPickle.load(open("data/ready/encoder-decoder-model-training/target_embedding_dict.pickle", "rb"))
idx_to_ent = {v: k for k, v in ent_to_idx.items()}

source_embedding_matrix = cPickle.load(open("data/ready/encoder-decoder-model-training/source_embedding_matrix.pickle", "rb"), encoding="latin1")
target_embedding_matrix = cPickle.load(open("data/ready/encoder-decoder-model-training/target_embedding_matrix.pickle", "rb"), encoding="latin1")

#以下是用来训练或者验证的数据集
train_dataset = cPickle.load(open("data/ready/encoder-decoder-model-training/train_dataset.pickle", "rb"))
test_dataset = cPickle.load(open("data/ready/encoder-decoder-model-training/test_dataset.pickle", "rb"))
val_dataset = cPickle.load(open("data/ready/encoder-decoder-model-training/val_dataset.pickle", "rb"))
trip_dataset = cPickle.load(open("data/ready/encoder-decoder-model-training/trip_dataset.pickle", "rb"))

pred_dict = cPickle.load(open("data/ready/pred_to_idx.pickle", "rb"))


# In[93]:


dictionary = cPickle.load(open("data/ready/dictionary.pickle","rb"), encoding="latin1")
wikidata_to_dbp = cPickle.load(open("data/ready/wikidata_link.pickle","rb"), encoding="latin1")  #wiki百科的链接数据集


# In[95]:


import string

#以下对数据集进行操作
training_input = list()
training_output = list()

for t in train_dataset:
    input_data, output_data, ner_data = t    #训练集中有三种数据，输入和输出数据都来自于此，训练集已经做好标志
    
    input_data = input_data.split()
    input_data = [word_to_idx[x] for x in input_data]    #设置输入数据集
    input_data.insert(0, word_to_idx["<START>"])   #设置开始和结束标志
    input_data.append(word_to_idx["<END>"])
    sent_array = [0] * INPUT_LENGTH             #创建一个数组用来存放数据集，
    sent_array[:len(input_data)] = input_data   #复制数据集里的内容至数组里面
    training_input.append(sent_array)           #将数组的内容添加到训练数据变量中，用来训来时使用：training_input
    
    output_data = [ent_to_idx[x] for x in output_data]   #设置输出数据集
    output_data.insert(0, ent_to_idx["<START>"])    #同样设置标志
    output_data.append(ent_to_idx["<END>"])
    target_array = [0] * OUTPUT_LENGTH
    target_array[:len(output_data)] = output_data
    training_output.append(target_array)
    
testing_input = list()
testing_output = list()
for t in test_dataset:       #训练数据和测试数据作同样的操作，训练数据和测试数据个是一模一样
    input_data, output_data, ner_data = t              #训练数据包含三个部分：原文字，输出序列，嵌入数据
    
    input_data = input_data.split()
    input_data = [word_to_idx[x] for x in input_data]
    input_data.insert(0, word_to_idx["<START>"])
    input_data.append(word_to_idx["<END>"])
    sent_array = [0] * INPUT_LENGTH
    sent_array[:len(input_data)] = input_data
    testing_input.append(sent_array)

    output_data = [ent_to_idx[x] for x in output_data]
    output_data.insert(0, ent_to_idx["<START>"])
    output_data.append(ent_to_idx["<END>"])
    target_array = [0] * OUTPUT_LENGTH
    target_array[:len(output_data)] = output_data
    testing_output.append(target_array)
    
    
validation_input = list()
validation_output = list()
for t in val_dataset:           #验证集也作同样的操作
    input_data, output_data, ner_data = t
    
    input_data = input_data.split()
    input_data = [word_to_idx[x] for x in input_data]
    input_data.insert(0, word_to_idx["<START>"])
    input_data.append(word_to_idx["<END>"])
    sent_array = [0] * INPUT_LENGTH
    sent_array[:len(input_data)] = input_data
    validation_input.append(sent_array)
    
    output_data = [ent_to_idx[x] for x in output_data]
    output_data.insert(0, ent_to_idx["<START>"])
    output_data.append(ent_to_idx["<END>"])
    target_array = [0] * OUTPUT_LENGTH
    target_array[:len(output_data)] = output_data
    validation_output.append(target_array)
    
trip_input = list()
trip_output = list()
for t in trip_dataset:        #trip同样的操作
    input_data, output_data, ner_data = t
    
    input_data = input_data.split()
    input_data = [word_to_idx[x] for x in input_data]
    input_data.insert(0, word_to_idx["<START>"])
    input_data.append(word_to_idx["<END>"])
    sent_array = [0] * INPUT_LENGTH
    sent_array[:len(input_data)] = input_data
    trip_input.append(sent_array)
    
    output_data = [ent_to_idx[x] for x in output_data]
    output_data.insert(0, ent_to_idx["<START>"])
    output_data.append(ent_to_idx["<END>"])
    target_array = [0] * OUTPUT_LENGTH
    target_array[:len(output_data)] = output_data
    trip_output.append(target_array)
    
# 以下就是论文汇总提到的co-conferece共参考操作
name_training_input = list()
name_training_output = list()
for e in ent_to_idx:        #target_embedding中的元素
    if e in wikidata_to_dbp:    #判断是否在wiki百科的实体数据库中。因wikidb有着完整的实体描述
        ent_name = wikidata_to_dbp[e].split("/")[-1].replace("_", " ").split()  #从wiki百科中获取实体
        in_embedding = True
        for token in ent_name:          #判断该实体是否已经做emdedding
            if token not in word_to_idx:
                in_embedding = False
        if not in_embedding:            #如果没有作embedding
            in_embedding = True
            ent_name = dictionary[e].split()    #从词典当中获取该实体的embedding
            for token in ent_name:              #对于该embeeding后的实体
                if token not in word_to_idx:    #如果不在source_emddeing中
                    in_embedding = False        #不再Source中，说明该embedding无效，即这是不属于使用范围的实体
        if in_embedding:                        #对于确定的embedding后的实体来说，对其对数据的适配操作
            source = list()                           #变量存储embedding后的wikidb中实体
            source.append(word_to_idx["<START>"])     #添加embedding形式的开始标志
            for token in ent_name:                    #将该实体添加进
                source.append(word_to_idx[token])
            source.append(word_to_idx["<END>"])
            target = [ent_to_idx["<START>"], ent_to_idx[e], ent_to_idx["<END>"]]  #同上source的操作这里用的实体的embedding,而不是word
            source[len(source):INPUT_LENGTH] = [0] * (INPUT_LENGTH-len(source))
            target[len(target):OUTPUT_LENGTH] = [0] * (OUTPUT_LENGTH-len(target))
            name_training_input.append(source)
            name_training_output.append(target)
    else:
        try:
            ent_name = dictionary[e].split()    #没有在wiki百科中的实体，则直接从字典中获取embedding。问题：与训练数据的关联
            in_embedding = True
            for token in ent_name:
                if token not in word_to_idx:
                    in_embedding = False
            if in_embedding:
                source = list()
                source.append(word_to_idx["<START>"])  
                for token in ent_name:
                    source.append(word_to_idx[token])
                source.append(word_to_idx["<END>"])
                target = [ent_to_idx["<START>"], ent_to_idx[e], ent_to_idx["<END>"]]
                source[len(source):INPUT_LENGTH] = [0] * (INPUT_LENGTH-len(source))
                target[len(target):OUTPUT_LENGTH] = [0] * (OUTPUT_LENGTH-len(target))
                name_training_input.append(source)
                name_training_output.append(target)
        except:
            pass

print('================================================================================')

print('word_to_idx【<START>】：',ent_to_idx["<START>"])
print('word_to_idx【<END>】：',word_to_idx["<END>"])
print('word_to_idx【team】：',word_to_idx["Born"])
print('idx_to_word【team】：',idx_to_word[249799])
#print('idx_to_word【team】：',idx_to_word[280326])
#print('idx_to_word【team】：',idx_to_word[16534])
print('idx_to_ent【?】：',idx_to_ent[249799],idx_to_ent[280326],idx_to_ent[16534])

print('ent_to_idx【<START>】：',ent_to_idx["<START>"])
print('ent_to_idx【<END>】：',ent_to_idx["<END>"])
print('ent_to_idx【280326】：',ent_to_idx["P19"])

print('pred_dict[P1040] ：',pred_dict['P1040'])
print('idx_to_ent[pred_dict[P1040]] ：',idx_to_ent[pred_dict['P1040']])
print('idx_to_word[pred_dict[P1040]] ：',idx_to_word[pred_dict['P1040']])
#'Q547625', 'P19', 'Q365'
print('dictionary[P1040] ：',dictionary['P1040'])
print('dictionary[Q547625] ：',dictionary['Q547625'])
print('dictionary[P19] ：',dictionary['P19'])
print('dictionary[Q365] ：',dictionary['Q365'])
#print('pred_dict ：',pred_dict)

print('len(train_dataset): ',len(train_dataset))
print('train_dataset[0][0]: ',train_dataset[0][0])

'''
fo = open('new_animal')
lines = fo.readlines()
count_animal = 0
animal_dataset=[]
for td in train_dataset:
    juzi = str(td[0])
    juzi = juzi.lower()
    #if i < 3:
    #    print(juzi)
    for line in lines:
        if line.strip('\n') in juzi:
            #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  :',i)
            count_animal = count_animal+1
            animal_dataset.append(td)
            break
            
cPickle.dump(animal_dataset, open("animal_dataset.pickle","wb"))
#cPickle.dump(lis, open("list.pkl","w"))
print('animal_dataset长度：',len(animal_dataset))
print('总计粗匹配出的句子有：',count_animal)
fo.close()
'''


i=0
j=6
for t in train_dataset:
    if i<j:
        print('train_dataset的数据格式：',t)
        print('长度：',len(t))
    i=i+1
i=0
for t in training_input:
    if i<j:
        print('training_input的数据格式：',t)
        print('长度：',len(t))
    i=i+1
i=0
for t in training_output:
    if i<j:
        print('training_output的数据格式：',t)
        print('长度：',len(t))
    i=i+1        
for t in training_output:
    if i<j:
        print('name_training_input的数据格式：',t)
        print('长度：',len(t))
    i=i+1  
for t in training_output:
    if i<j:
        print('name_training_output的数据格式：',t)
        print('长度：',len(t))
        i=i+1      

print('================================================================================')


print ('training size 0 ：', len(training_input))  
print ('name_training_input 0 ：', len(name_training_input)) 
name_training_input += training_input      #与训练数据的关联在此，增添，重置。问题：且只有训练数据，无测试数据等，为何这般折腾？
name_training_output += training_output    #这般折腾相当于使用ent_embedding和wikidb给训练集打标签
training_input = name_training_input       #上面的word_emddeing适用于于原始数据，ent_emddeing相当于打标签后的数据，可做训练使用
training_output = name_training_output     #因此，实际上只对训练数据作共参考操作

#以下对数据作适应性的转码，因为很可能不可直接使用
training_input = array(training_input, dtype=np.int32)
training_output = array(training_output, dtype=np.int32)
testing_input = array(testing_input, dtype=np.int32)
testing_output = array(testing_output, dtype=np.int32)
validation_input = array(validation_input, dtype=np.int32)
validation_output = array(validation_output, dtype=np.int32)
trip_input = array(trip_input, dtype=np.int32)
trip_output = array(trip_output, dtype=np.int32)

'''
print ('training size', len(training_input))
print ('validation size', len(validation_input))
print ('test size', len(testing_input))
print ('trip size', len(trip_input))
'''


# #### Build MODEL

# In[82]:


BUFFER_SIZE = len(training_input)
#BATCH_SIZE = 64
BATCH_SIZE = 16
N_BATCH = BUFFER_SIZE//BATCH_SIZE
#embedding_dim = 64
embedding_dim = 64
#units = 512      #GRU的门数量
units = 256
vocab_inp_size = len(word_to_idx)
vocab_tar_size = len(ent_to_idx)

#读取数据集,训练集是已经做过扩充的
train_dataset = tf.data.Dataset.from_tensor_slices((training_input, training_output)).shuffle(BUFFER_SIZE)
#train_dataset = train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
train_dataset = train_dataset.batch(batch_size=BATCH_SIZE, drop_remainder=True)   #2.0版本
val_dataset = tf.data.Dataset.from_tensor_slices((validation_input, validation_output)).shuffle(BUFFER_SIZE)
#val_dataset = val_dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE)) 
val_dataset = val_dataset.batch(batch_size=BATCH_SIZE, drop_remainder=True)   #2.0版本


# In[83]:


#GRU(Gated recurrent unit)。是LSTM的简化版本，是LSTM的变体，它去除掉了细胞状态，使用隐藏状态来进行信息的传递
#tensorfow2.0中将这三个封装到以下接口中：keras.layers.SimpleRNN,keras.layers.GRU,keras.layers.LSTM.默认使用cudnn加速

def gru(units):
    if tf.test.is_gpu_available():     #使用cudnn加速
        #return tf.keras.layers.CuDNNGRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        return tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_activation='sigmoid',recurrent_initializer='glorot_uniform',reset_after = True)   #2.0

    else:
        return tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')


# In[84]:


#### ENCODER
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embeddings_initializer = tf.constant_initializer(source_embedding_matrix)
        #self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, trainable=True)
        #删除mask，tf2.0不支持mask
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=False, trainable=True)
        
        self.gru = gru(self.enc_units)
        
    def get_ngram_tensor(self, stacked_tensor, N):
        unstacked_tensor = tf.unstack(stacked_tensor, stacked_tensor.get_shape().as_list()[1], 1)
        result = list()
        for i in range(len(unstacked_tensor)-(N-1)):
            tmp_tensor = unstacked_tensor[i]
            for j in range(1, N):
                tmp_tensor = tf.concat([tmp_tensor, unstacked_tensor[j+i]], 1)
            result.append(tmp_tensor)
        result = tf.stack(result, 1)
        return result
    
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        
        embedded_input = x
        return output, state, embedded_input
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


# In[85]:


#### DECODER
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embeddings_initializer = tf.constant_initializer(target_embedding_matrix)
        #self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, trainable=True)
        #删除mask
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=False, trainable=True)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
        self.W1_1 = tf.keras.layers.Dense(self.dec_units)
        self.V_1 = tf.keras.layers.Dense(1)
        
        self.W1_2 = tf.keras.layers.Dense(self.dec_units)
        self.V_2 = tf.keras.layers.Dense(1)
        
        self.W1_3 = tf.keras.layers.Dense(self.dec_units)
        self.V_3 = tf.keras.layers.Dense(1)
        
        self.W_cv1 = tf.keras.layers.Dense(self.dec_units)
        self.W_cv2 = tf.keras.layers.Dense(self.dec_units)
        self.W_cv3 = tf.keras.layers.Dense(self.dec_units)
        
    def get_ngram_tensor(self, stacked_tensor, N):
        unstacked_tensor = tf.unstack(stacked_tensor, stacked_tensor.get_shape().as_list()[1], 1)
        result = list()
        for i in range(len(unstacked_tensor)-(N-1)):
            tmp_tensor = unstacked_tensor[i]
            for j in range(1, N):
                tmp_tensor = tf.concat([tmp_tensor, unstacked_tensor[j+i]], 1)
            result.append(tmp_tensor)
        result = tf.stack(result, 1)
        return result
        
    def call(self, x, hidden, enc_output, embedded_input):
        
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        x = self.embedding(x)
        
        enc_output_1 = self.get_ngram_tensor(embedded_input, 1)
        score_1 = self.V_1(tf.nn.tanh(self.W1_1(enc_output_1) + self.W2(hidden_with_time_axis)))
        attention_weights_1 = tf.nn.softmax(score_1, axis=1)
        context_vector_1 = attention_weights_1 * enc_output_1
        context_vector_1 = tf.reduce_sum(context_vector_1, axis=1)
        context_vector_1 = self.W_cv1(context_vector_1)

        enc_output_2 = self.get_ngram_tensor(embedded_input, 2)
        score_2 = self.V_2(tf.nn.tanh(self.W1_2(enc_output_2) + self.W2(hidden_with_time_axis)))
        attention_weights_2 = tf.nn.softmax(score_2, axis=1)
        context_vector_2 = attention_weights_2 * enc_output_2
        context_vector_2 = tf.reduce_sum(context_vector_2, axis=1)
        context_vector_2 = self.W_cv2(context_vector_2)

        enc_output_3 = self.get_ngram_tensor(embedded_input, 3)
        score_3 = self.V_3(tf.nn.tanh(self.W1_3(enc_output_3) + self.W2(hidden_with_time_axis)))
        attention_weights_3 = tf.nn.softmax(score_3, axis=1)
        context_vector_3 = attention_weights_3 * enc_output_3
        context_vector_3 = tf.reduce_sum(context_vector_3, axis=1)
        context_vector_3 = self.W_cv3(context_vector_3)

        context_vector_n = tf.nn.tanh(context_vector_1 + context_vector_2 + context_vector_3)

        x = tf.concat([tf.expand_dims(context_vector_n, 1), tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        
        return x, state, attention_weights
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


# In[86]:


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)


# In[87]:


#optimizer = tf.train.AdamOptimizer(0.0002)
optimizer = tf.optimizers.Adam(0.0002)   #2.0

#定义损失函数
def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


# In[88]:


checkpoint_dir = 'data/ready/encoder-decoder-model-training/model/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


# In[89]:


EPOCHS = TOTAL_EPOCHS  #20
#EPOCHS = 2            #新增

#log_file = open("log", "w", 0)
log_file = open("log", "w") #2.0

for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(train_dataset):
        hidden = encoder.initialize_hidden_state()
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden, embedded_input = encoder(inp, hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([word_to_idx['<START>']] * BATCH_SIZE, 1)       

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, embedded_input)
                loss += loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        total_loss += batch_loss
        variables = encoder.variables + decoder.variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        if batch % 100 == 0:
            log_str = ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
            print (log_str)
            log_file.write(log_str+"\n")
    
    #if (epoch + 1) % 10 == 0:    #每十代保存一次检查点
    if (epoch + 1) % 5 == 0:    #修改成5代保存
        checkpoint.save(file_prefix = checkpoint_prefix)

    val_loss = 0
    for (batch, (inp, targ)) in enumerate(val_dataset):
        hidden = encoder.initialize_hidden_state()
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden, embedded_input = encoder(inp, hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([word_to_idx['<START>']] * BATCH_SIZE, 1)       

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, embedded_input)
                loss += loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = (loss / int(targ.shape[1]))
        val_loss += batch_loss

    log_str = ('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / N_BATCH))
    print (log_str)
    log_file.write(log_str+"\n")
    log_str = ('Epoch {} Val Loss {:.4f}'.format(epoch + 1, val_loss / (len(validation_input)//BATCH_SIZE)))
    print (log_str)
    log_file.write(log_str+"\n")
    log_str = ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print (log_str)
    log_file.write(log_str+"\n")
print ("DONE")
log_file.write("DONE\n")
log_file.close()


# #### Testing

# In[14]:


# LOAD EXISTING MODEL
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# In[15]:


def get_gold_standard(expected_output):
    result = []
    for e in expected_output:
        result.append(idx_to_ent[e])
    return result


# In[16]:


#将学习后的checkpoint经过三元组分类器，输出成三元组，学习的结果是实体和谓词ID的对应
clf = cPickle.load(open("data/ready/triple_classifier_model/model.pickle", "rb"), encoding="latin1")    #2.0

def one_hot_encoding(pred, pred_dict):
    enc = [0]*len(pred_dict)
    enc[pred_dict[pred]] = 1
    return enc

def get_sent(result, prob_result, postprocessing):
    sent_set = set()
    sent = list()
    sent_id = list()
    for i,token in enumerate(result):
        if token == "<START>":
            continue
        if token == "<END>":
            if len(sent) == 3:
                sent_set.add(" ### ".join(sent))
            return sent_set
        sent.append(dictionary[token])
        sent_id.append(token)
        if len(sent) == 3:
            if postprocessing:
                if (prob_result[i]+prob_result[i-1]+prob_result[i-2])/3 < 0.70:
                    s = sent_id[0]
                    p = sent_id[1]
                    o = sent_id[2]
                    
                    #Q是实体，P是关系
                    if s.startswith("Q") and p.startswith("P") and o.startswith("Q"):

                        embedding_s = np.asarray(target_embedding_matrix[ent_to_idx[s]]).astype(np.float)
                        embedding_o = np.asarray(target_embedding_matrix[ent_to_idx[o]]).astype(np.float)
                        embedding_p = np.asarray(target_embedding_matrix[ent_to_idx[p]]).astype(np.float)
                        diff = (abs(embedding_s+embedding_p-embedding_o)).sum()
                        valid_triple = clf.predict([one_hot_encoding(p, pred_dict)+ one_hot_encoding(TEST_DATASET, {"seen":0,"nyt":1,"trip":2})+[diff]])[0]
                        if valid_triple == 0:
                            sent_set.add(" ### ".join(sent))
                else:
                    sent_set.add(" ### ".join(sent))
            else:
                sent_set.add(" ### ".join(sent))
            sent = list()
    return sent_set


# In[17]:


def get_sent_from_training_data(training_data):
    sent = ""
    for tok in training_data:
        sent += idx_to_word[tok]+" "
        if idx_to_word[tok] == "<END>":
            return sent.strip() 
    return sent.strip()


# In[44]:


from kitchen.text.converters import getwriter, to_bytes, to_unicode
from kitchen.i18n import get_translation_object
translations = get_translation_object('example')
_ = translations.ugettext
b_ = translations.lgettext

def phrase_sim(sent, phrases):
    result = list()
    pos = 0
    for p in phrases:
        l = len(p.split())
        splitted = sent.split()
        max_score = 0
        for i in range(len(splitted)-l+1):
            test = " ".join(splitted[i:i+l])
            score = fuzz.ratio(p, test)
            if score > max_score:
                max_score = score
        result.append((p, len(phrases)-pos, max_score))
        pos += 1
    result = sorted(result, key = lambda x: (x[2], x[1],len(x[0])), reverse=True)
    return result

def generate_output(sentence, encoder, decoder):
    attention_plot = np.zeros((OUTPUT_LENGTH, INPUT_LENGTH))
    sentence_ori = get_sent_from_training_data(sentence[0])
    result = list()
    prob_result = list()
    inputs = tf.convert_to_tensor(sentence)
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden, embedded_input = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([ent_to_idx['<START>']], 0)

    if BEAMSEARCH:
        for t in range(OUTPUT_LENGTH):
            if t%3==0 or t%3==2:
                beam = True
            else:
                beam = False
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out, embedded_input)
            prob_predictions = tf.nn.softmax(predictions)
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            if prob_predictions[0][predicted_id] > 0.99:
                beam = False
            
            if idx_to_ent[predicted_id] == '<END>':
                return result,prob_result,sentence_ori, attention_plot

            if beam:
                if t%3==2:
                    emb_s = np.asarray(target_embedding_matrix[ent_to_idx[result[t-2]]])
                    emb_p = np.asarray(target_embedding_matrix[ent_to_idx[result[t-1]]])
                    diff = tf.cast(tf.nn.softmax(-np.sum(abs(emb_s+emb_p-target_embedding_matrix), axis=1), axis=0), tf.float32)
                    prob_predictions_tmp = tf.multiply(prob_predictions[0], diff)
                    cand_predicted_id = tf.nn.top_k(prob_predictions_tmp, k=100, sorted=True)[1].numpy()
                else:
                    cand_predicted_id = tf.nn.top_k(predictions[0], k=10, sorted=True)[1].numpy()
                
                phrases = list()
                rev_dict = dict()
                for p in cand_predicted_id:
                    if idx_to_ent[p] == "<END>" or idx_to_ent[p] == "<START>":
                        continue
                    phrases.append(_(dictionary[idx_to_ent[p]]))
                    rev_dict[dictionary[idx_to_ent[p]]] = idx_to_ent[p]
                #predicted_id = np.int64(ent_to_idx[rev_dict[b_(phrase_sim(sentence_ori, phrases)[0][0])]])
                predicted_id = np.int64(ent_to_idx[rev_dict[_(phrase_sim(sentence_ori, phrases)[0][0])]])  #tf2.0
            else:
                predicted_id = tf.argmax(predictions[0]).numpy()
                
            result.append(idx_to_ent[predicted_id])
            prob_result.append(prob_predictions[0][predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)
    else:
        for t in range(OUTPUT_LENGTH):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out, embedded_input)
            prob_predictions = tf.nn.softmax(predictions)
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()
            
            predicted_id = tf.argmax(predictions[0]).numpy()

            if idx_to_ent[predicted_id] == '<END>':
                return result,prob_result,sentence_ori, attention_plot
            
            result.append(idx_to_ent[predicted_id])
            prob_result.append(prob_predictions[0][predicted_id])
            dec_input = tf.expand_dims([predicted_id], 0)

    return result,prob_result,sentence_ori, attention_plot


# In[45]:


INSPECT = True
TEST_DATASET = "seen" # 'trip' or 'seen'

log_file = open("result_log", "w")

if TEST_DATASET == "nyt":
    test_in = nyt_input
    test_out = nyt_output
elif TEST_DATASET == "seen":
    test_in = testing_input
    test_out = testing_output
elif TEST_DATASET == "trip":
    test_in = trip_input
    test_out = trip_output

if INSPECT:
    len_test = 100
else:
    len_test = len(test_in)

num_correct = 0
total_prediction = 0
total_gold_standard = 0
eval_dataset = (test_in[:len_test], test_out[:len_test])
for i, test_sent in enumerate(eval_dataset[0]):
    if (i+1) % 100 == 0:
        print (i,str(num_correct/float(total_prediction)),str(num_correct/float(total_gold_standard)))
        log_file.write(str(i)+" "+str(num_correct/float(total_prediction))+" "+str(num_correct/float(total_gold_standard))+"\n")
    result,prob_result,sentence, attention_plot = generate_output([test_sent], encoder, decoder)
    result_sent = get_sent(result[:], prob_result, POSPROCESSING)
    gold_standard = get_gold_standard(eval_dataset[1][i])
    gold_standard_sent = get_sent(gold_standard,None, False)


    if INSPECT:
        if result_sent != gold_standard_sent:
            print (i)
            print ("Input: ", sentence)
            print ("Predicted: ", result, result_sent)
            print ("Expected: ", gold_standard, gold_standard_sent)
        if i > 100:
            break

    total_prediction += len(result_sent)
    total_gold_standard += len(gold_standard_sent)

    for s in result_sent:
        if s in gold_standard_sent:
            num_correct += 1
print ("PRECISION: ", num_correct/float(total_prediction))
print ("RECALL: ", num_correct/float(total_gold_standard))
log_file.write("PRECISION: "+str(num_correct/float(total_prediction))+"\n")
log_file.write("RECALL: "+str(num_correct/float(total_gold_standard))+"\n")


# In[ ]:





# In[ ]:




