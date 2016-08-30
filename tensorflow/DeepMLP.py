import sys
import time
import re
import collections
import numpy
import tensorflow as tf

def readHeader(f):
    rawHeader = next(f)[:-1].split(',')

    #group columns per stations
    groupHeader = collections.OrderedDict()
    pattern = re.compile("L(\d)_S(\d+)")    
    for i, h in enumerate(rawHeader):
        m = pattern.search(h)
        if(m==None):
            groupHeader[h] = [i]
        elif(m.group(0) not in groupHeader):
            groupHeader[m.group(0)]  = [i]
        else:
            groupHeader[m.group(0)] += [i]

    #reduce the list of index to a range
    for k in groupHeader:
        groupHeader[k] =  (numpy.amin(groupHeader[k]) , numpy.amax(groupHeader[k])+1)

    return groupHeader

def castFloat(s):
    if(s==''):return float(0)
    return float(s)

def readBatch(f, BatchSize, groupHeader, train=False, test=False, validation=False):
   data = collections.OrderedDict()
   #initialize data
   for g in groupHeader:
       data[g] = []

   EOF = False
   while(len(data["Id"])<BatchSize and not EOF):
      #read a new line in the file
      row = next(f)
      EOF |= row[-1]!="\n"
      if(not EOF): row = row[:-1]
      row = row.split(',')

      #filter
      testFlag = (float(row[groupHeader["Response"][0]])==-1)
      valFlag  = (int(row[groupHeader["Id"][0]])    %100==0)
      if(train      and (    testFlag or     valFlag) ): continue
      if(test       and (not testFlag or     valFlag) ): continue
      if(validation and (    testFlag or not valFlag) ): continue
   
      #change datatype
      for g in groupHeader:
          i,j   = groupHeader[g]
          value = row[i:j]
          if   g == "Id":       value = [int(l) for l in value]
          elif g == "Pattern":  value = [[int(l) for l in list(value[0])]]
          elif g == "Response": value = [castFloat(l) for l in value]
          else : value = [ castFloat(l) for l in value  ]

          data[g] += value
          
   return data




with open('../normalized.csv', 'r') as f:
   groupHeader = readHeader(f)

   batch = readBatch(f, 1000, groupHeader, validation=True)
   for b in batch:
      print(b)


x_ = tf.placeholder(tf.float32, shape=[4,2], name = 'x-input')
y_ = tf.placeholder(tf.float32, shape=[4,1], name = 'y-input')

Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name = "Theta1")
Theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name = "Theta2")

Bias1 = tf.Variable(tf.zeros([2]), name = "Bias1")
Bias2 = tf.Variable(tf.zeros([1]), name = "Bias2")

with tf.name_scope("layer2") as scope:
	A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)

with tf.name_scope("layer3") as scope:
	Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

with tf.name_scope("cost") as scope:
	cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + 
		((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)

with tf.name_scope("train") as scope:
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.initialize_all_variables()
sess = tf.Session()

writer = tf.train.SummaryWriter("./logs/xor_logs", sess.graph_def)

sess.run(init)

t_start = time.clock()
for i in range(100000):
	sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
	if i % 1000 == 0:
		print('Epoch ', i)
		print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
		print('Theta1 ', sess.run(Theta1))
		print('Bias1 ', sess.run(Bias1))
		print('Theta2 ', sess.run(Theta2))
		print('Bias2 ', sess.run(Bias2))
		print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
t_end = time.clock()
print('Elapsed time ', t_end - t_start)







