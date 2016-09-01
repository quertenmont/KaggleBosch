import sys
import math
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
    if(s==''):return float(-1)
    return float(s)

def readBatch(f, BatchSize, groupHeader, train=False, test=False, validation=False):
   data = collections.OrderedDict()
   #initialize data
   for g in groupHeader:
       data[g] = []

   background = 0

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

          if(train and background>int(BatchSize*0.2) and float(row[groupHeader["Response"][0]])==0):continue
          if(float(row[groupHeader["Response"][0]])==0):background+=1
       
          #change datatype
          for g in groupHeader:
              i,j   = groupHeader[g]
              value = row[i:j]
              if   g == "Id":       value = [int(l) for l in value]
              elif g == "Pattern":  value = [int(l) for l in list(value[0])]
              elif g == "Response": value = [castFloat(l) for l in value]
              else : value = [ castFloat(l) for l in value  ]

              data[g] += [value]
      except:
          f.seek(0)
          break
          
   return data


def denseLayer(name, input, nIn, nOut):
    with tf.name_scope(name + "_"+str(nIn)+"to"+str(nOut)) as scope:
       W = tf.Variable(tf.truncated_normal([nIn,nOut], stddev=math.sqrt(3 / (nIn+nOut))), name = "weight")
       b = tf.Variable(tf.random_uniform([nOut], -0.1,0.1), name = "bias")
       return tf.sigmoid(tf.matmul(input, W) + b)

def deepMLP(input, layers):
    results = input
    for i in range(1,len(layers)):
        results = denseLayer("layer"+str(i), results, layers[i-1], layers[i])
    return results #we want a probability in [0 , 1]

#global variable
nameToTFVariable = {}
def stationNN(station, groupHeader, Pattern, index):
    global nameToTFVariable
    
    nOut  = 1
    nIn   = groupHeader[station][1] - groupHeader[station][0]
    input = tf.placeholder(tf.float32, shape=[None,nIn], name = station+"-input")    
    nameToTFVariable[station] = input

    with tf.name_scope(station) as scope:
       out = deepMLP(input, [int(nIn), int(math.ceil(nIn*4/3)), int(math.ceil(nIn*2/3)),  int(nOut)])  #=> OUT OF MEMORY
#       out = deepMLP(input, [int(nIn), int(nIn*4/3), int(nOut)])

       with tf.name_scope("Gate") as scope:
          gate = tf.expand_dims(Pattern[:,index],1)
          gated = (gate*(out+1)) - 1          
       return gated



def getFeedDict(data):
    feedDict = {}
    for n in nameToTFVariable:
#        if(n=="L0_S05"):print(data[n])
        feedDict[nameToTFVariable[n]] = data[n]
    return feedDict


def loss_log(out, target):
    return - tf.reduce_sum( target * tf.log(out)  + (1.0-target)*tf.log(1.0-out) )

def loss_hinge(out, target):
    return tf.reduce_sum( tf.maximum( 0.0, 1.0 - ((2.0*target)-1.0) * ((2.0*out)-1.0)  ) )    


###############MAIN#####################

batchSize = 400

GroupList = ['L0_S00','L0_S01','L0_S02','L0_S03','L0_S04','L0_S05','L0_S06','L0_S07','L0_S08','L0_S09','L0_S10','L0_S11','L0_S12','L0_S13','L0_S14','L0_S15','L0_S16','L0_S17','L0_S18','L0_S19','L0_S20','L0_S21','L0_S22','L0_S23','L1_S24','L1_S25','L2_S26','L2_S27','L2_S28','L3_S29','L3_S30','L3_S31','L3_S32','L3_S33','L3_S34','L3_S35','L3_S36','L3_S37','L3_S38','L3_S39','L3_S40','L3_S41','L3_S42','L3_S43','L3_S44','L3_S45','L3_S46','L3_S47','L3_S48','L3_S49','L3_S50','L3_S51']

#read the header and the validation sample
with open('../normalized.csv', 'r') as f:
   groupHeader = readHeader(f)
   validation = readBatch(f, batchSize, groupHeader, validation=True)
   #for g in groupHeader:   print(g)

#build the graph of the neural network
graph = tf.Graph()
with graph.as_default():
    #prepare the optimizer that we can use altrough
    with tf.name_scope("trainer") as scope:
        optimizer = tf.train.RMSPropOptimizer(0.1)#, use_locking=True)


    #load variables
    Response = tf.placeholder(tf.float32, shape=[None,1], name = 'Response-input')
    nameToTFVariable["Response"] = Response

    with tf.name_scope("PerStationMLP") as scope:
        Pattern = tf.placeholder(tf.float32, shape=[None,52], name = 'Pattern-gate')    
        nameToTFVariable["Pattern"] = Pattern

        outAtStation =[]
        pretrainingAtStation = []
        for i,g in enumerate(GroupList):
           out =  stationNN(g, groupHeader, Pattern, i)
           outAtStation += [out]

           with tf.name_scope("pre-training") as scope:
              with tf.name_scope("loss") as scope:      
                 patternMask = tf.greater(Pattern[:,i],0.5)
                 stationloss = loss_hinge(tf.boolean_mask(out, patternMask), tf.boolean_mask(Response, patternMask))
                 stationcost = stationloss
                 
              with tf.name_scope("trainer") as scope:
                  training_var = []
                  for v in tf.trainable_variables() :
                      if(g in v.name):training_var += [v]
                  grads = optimizer.compute_gradients(stationcost, var_list=training_var )#, var_list=training_var)

                  pretrainingAtStation += [ optimizer.minimize(stationcost) ]
#        pretraining = tf.group(pretrainingAtStation, name="pre-training")                 


           

    #recombine the output of each station into one single MLP
    with tf.name_scope("RecombinationMLP") as scope:
        outAtStationPack = tf.concat(1,outAtStation)
        nStations = int(len(GroupList))
        outFinal = deepMLP(outAtStationPack, [nStations, int(math.ceil(nStations*4/3)), int(math.ceil(nStations*2/3)), int(1)])

#        for v in tf.trainable_variables(): print("trainame = " +v.name + " --> " + str("L0_S21" in v.name))
#        for v in filter(lambda x:"L0_S21" in x.name, tf.trainable_variables()): print("trainame = " +v.name)

    with tf.name_scope("cost") as scope:
        with tf.name_scope("L2") as scope:            
           L2   = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 0.00001

        with tf.name_scope("evaluation") as scope:
           TP  = 0.0001 + tf.reduce_sum( tf.cast(tf.logical_and(tf.greater_equal(Response, 0.5) , tf.greater_equal(outFinal, 0.5) ), tf.float32) )
           TN  = 0.0001 + tf.reduce_sum( tf.cast(tf.logical_and(tf.         less(Response, 0.5) , tf.         less(outFinal, 0.5) ), tf.float32) )
           FP  = 0.0001 + tf.reduce_sum( tf.cast(tf.logical_and(tf.         less(Response, 0.5) , tf.greater_equal(outFinal, 0.5) ), tf.float32) )
           FN  = 0.0001 + tf.reduce_sum( tf.cast(tf.logical_and(tf.greater_equal(Response, 0.5) , tf.         less(outFinal, 0.5) ), tf.float32) )
           MCC = tf.div ( tf.sub(tf.mul(TP,TN),tf.mul(FP,FN)) , tf.sqrt( tf.mul( tf.mul(tf.add(TP,FP),tf.add(TP,FN)) , tf.mul(tf.add(TN,FP),tf.add(TN,FN)) ) ) )

        with tf.name_scope("loss") as scope:      
#           debug = tf.maximum( 0.0, 1.0 - (2.0*(Response-1.0)) * (2.0*(outFinal-1.0))  )
           debug = 1.0 - ((2.0*Response)-1.0) * ((2.0*outFinal)-1.0)
           crossentropy = loss_hinge(outFinal, Response)
           cost = crossentropy# + L2
        #cost = tf.Print(crossentropy, [Response, outFinal, TP, TN, FN, FP, MCC,  TPa, TNa, FPa, FNa, MCCa])  # + L2


    with tf.name_scope("trainer") as scope:
        training = optimizer.minimize(cost)#, gate_gradients=optimizer.GATE_OP)

        #grads_and_vars = optimizer.compute_gradients(cost)
        #gradsA = tf.Print(grads_and_vars[0][0], grads_and_vars)

    with tf.name_scope("summary") as scope:        
        tf.scalar_summary('cost', cost)
        tf.scalar_summary('crossentropy', crossentropy)
        tf.scalar_summary('L2', L2)
        tf.scalar_summary('TP', TP)
        tf.scalar_summary('TN', TN)
        tf.scalar_summary('FP', FP)
        tf.scalar_summary('FN', FN)
        tf.scalar_summary('MCC', MCC)
        TFSummary = tf.merge_all_summaries()        


    init = tf.initialize_all_variables()
    saver        = tf.train.Saver()



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
            #saver.save(sess, "saves/pretraining.ckpt")
        else:
            totalLineRead += batchSize

        #train on the batch
        #grads = sess.run([gradsA], feed_dict=getFeedDict(batch))
        #print(grads)

#        gate = sess.run([gatesA], feed_dict=getFeedDict(batch))
#        print("Gates:"+str(gate))
#        print("Patterns:"+str(batch["Pattern"]))

        if(epoch<=100):
           #per station pretraining
           outTrain = sess.run([cost, TFSummary, outFinal, debug] + pretrainingAtStation, feed_dict=getFeedDict(batch))
        else:
           #global training
           outTrain = sess.run([cost, TFSummary, outFinal, debug, training], feed_dict=getFeedDict(batch))

        if iteration % 10 == 0:
           outValid = sess.run([cost, TFSummary], feed_dict=getFeedDict(validation))
           print("Iteration=%5i Epoch=%2i LineRead=%8i --> cost (train) = %6.3f / cost (validation) = %+6.3f" % (iteration,epoch,totalLineRead,outTrain[0], outValid[0]))

           compString = ""
           for i,pred in enumerate(outTrain[2]):
               if(i>10):break;
               compString += "%s vs %s (%s),  " % ( str(batch["Response"][i][0]), float(pred), (outTrain[3][i][0]) )
           print(compString)
           writer_train.add_summary(outTrain[1], totalLineRead)
           writer_val  .add_summary(outValid[1], totalLineRead)



        if(epoch>=200):break #stopping after nloop on the files
t_end = time.clock()
print('Elapsed time ', t_end - t_start)







