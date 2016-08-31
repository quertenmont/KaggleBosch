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
      try:
          #read a new line in the file
          row = next(f)
          EOF |= row[-1]!="\n"
          if(not EOF): row = row[:-1]
          row = row.split(',')

          #skip header
          if("Id" in row[0]):continue #this is the file header

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

              data[g] += [value]
      except:
          f.seek(0)
          break
          
   return data


def denseLayer(name, input, nIn, nOut):
#    with tf.name_scope(name + "_"+str(nIn)+"In_"+str(nOut)+"Out") as scope:
       W = tf.Variable(tf.truncated_normal([nIn,nOut]), name = "weight")
       b = tf.Variable(tf.zeros([nOut]), name = "bias")
       return tf.sigmoid(tf.matmul(input, W) + b)

def deepMLP(input, layers):
    results = input
    for i in range(1,len(layers)):
        results = denseLayer("MLPlayer"+str(i), results, layers[i-1], layers[i])
    return tf.nn.relu(results)

#global variable
nameToTFVariable = {}
def stationNN(station, groupHeader):
    global nameToTFVariable
    
    nOut  = 1
    nIn   = groupHeader[station][1] - groupHeader[station][0]
    input = tf.placeholder(tf.float32, shape=[None,nIn], name = station+"-input")    
    nameToTFVariable[station] = input
#    with tf.name_scope(station) as scope:
    #return deepMLP(input, [int(nIn), int(nIn*4/3), int(nIn*2/3), int(nOut)])
    return deepMLP(input, [int(nIn), int(nOut)])

        

def getFeedDict(data):
    feedDict = {}
    for n in nameToTFVariable:
#        if(n=="L0_S05"):print(data[n])
        feedDict[nameToTFVariable[n]] = data[n]
    return feedDict



###############MAIN#####################

batchSize = 2500

#read the header and the validation sample
with open('../normalized.csv', 'r') as f:
   groupHeader = readHeader(f)
   validation = readBatch(f, batchSize, groupHeader, validation=True)

#build the graph of the neural network
graph = tf.Graph()
with graph.as_default():
    Response = tf.placeholder(tf.float32, shape=[None,1], name = 'Response-input')
    nameToTFVariable["Response"] = Response
    
    with tf.name_scope("ML") as scope:
        out_L0_S05 = stationNN("L0_S05", groupHeader)

#        Hypothesis2 = stationNN("L0_S06", groupHeader)

        for v in tf.trainable_variables():
           print("trainame = " +v.name)


    with tf.name_scope("cost") as scope:
        #subtle gymnastic to keep everything differentiable
        TP = 0.1 + tf.reduce_sum( tf.select(tf.logical_and(tf.   greater(Response, 0.5), tf.   greater(out_L0_S05, 0.5)) , tf.maximum(tf.minimum(out_L0_S05,1.0),1.0), tf.maximum(tf.minimum(out_L0_S05,0.0),0.0)  ) )
        TN = 0.1 + tf.reduce_sum( tf.select(tf.logical_and(tf.less_equal(Response, 0.5), tf.less_equal(out_L0_S05, 0.5)) , tf.maximum(tf.minimum(out_L0_S05,1.0),1.0), tf.maximum(tf.minimum(out_L0_S05,0.0),0.0)  ) )
        FN = 0.1 + tf.reduce_sum( tf.select(tf.logical_and(tf.   greater(Response, 0.5), tf.less_equal(out_L0_S05, 0.5)) , tf.maximum(tf.minimum(out_L0_S05,1.0),1.0), tf.maximum(tf.minimum(out_L0_S05,0.0),0.0)  ) )
        FP = 0.1 + tf.reduce_sum( tf.select(tf.logical_and(tf.less_equal(Response, 0.5), tf.   greater(out_L0_S05, 0.5)) , tf.maximum(tf.minimum(out_L0_S05,1.0),1.0), tf.maximum(tf.minimum(out_L0_S05,0.0),0.0)  ) )
        MCC  = 1 - tf.div ( tf.sub(tf.mul(TP,TN),tf.mul(FP,FN)) ,  tf.sqrt( tf.mul( tf.mul(tf.add(TP,FP),tf.add(TP,FN)) , tf.mul(tf.add(TN,FP),tf.add(TN,FN)) ) ) )
        cost = MCC 

        tf.scalar_summary('cost', cost)
        tf.scalar_summary('MCC', MCC)
        tf.scalar_summary('TP', TP)
        tf.scalar_summary('TN', TN)

    with tf.name_scope("trainer") as scope:
        optimizer = tf.train.AdamOptimizer(0.01, use_locking=True)
        training = optimizer.minimize(cost)#, gate_gradients=optimizer.GATE_OP)


    TFSummary = tf.merge_all_summaries()        
    init = tf.initialize_all_variables()


#create the running session on the graph

#with tf.Session(graph=graph) as sess:
sess = tf.Session(graph=graph)
writer_train = tf.train.SummaryWriter("./tflogs/train", graph)
writer_val   = tf.train.SummaryWriter("./tflogs/validation", graph)
sess.run(init)

t_start = time.clock()
totalLineRead = 0
epoch = 0
iteration = 0
with open('../normalized.csv', 'r') as f:
    while(True): 
        iteration+=1
        #read a batch of train data
        batch = readBatch(f, batchSize, groupHeader, train=True)
        if(len(batch["Id"])!=batchSize): #read EOF or something went wrong
            f.seek(0)#rewind
            epoch         += 1
            totalLineRead += len(batch["Id"])
        else:
            totalLineRead += batchSize

        #train on the batch
        outTrain = sess.run([cost, TFSummary, training], feed_dict=getFeedDict(batch))
        if iteration % 1 == 0:
           outValid = sess.run([cost, TFSummary], feed_dict=getFeedDict(validation))
           print("Iteration=%5i Epoch=%2i LineRead=%8i --> cost (train) = %6.3f / cost (validation) = %+6.3f" % (iteration,epoch,totalLineRead,outTrain[0], outValid[0]))
           writer_train.add_summary(outTrain[1], totalLineRead)
           writer_val  .add_summary(outValid[1], totalLineRead)


        if(epoch>=10):break #stopping after nloop on the files
t_end = time.clock()
print('Elapsed time ', t_end - t_start)







