import numpy as np
import pandas as pd
import tensorflow as tf


dataset = pd.read_csv("data.csv")

dataset = dataset.drop(["RowNumber","UID","Customer_name","City","Gender"],axis=1)

features = dataset.iloc[:,:-1].values.astype(np.float32)
y_temp = dataset.iloc[:,[8]].values.astype(np.float32)

y = np.zeros((len(y_temp),2))

for ind,i in enumerate(y_temp):
    y[ind,int(i)]=1

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split( features , y , test_size=0.3)


n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100
n_nodes_hl4 = 100

n_classes = 2
batch_size = 100
pointer = 0

# PLACEHOLDERS
x = tf.placeholder(tf.float32,shape=[None,8])
y_true = tf.placeholder(tf.float32,shape = [None,2])


hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([8,n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

hidden_layer_4 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl4]))}

output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                  'bias': tf.Variable(tf.random_normal([n_classes]))}

#y=xW+b

l1 = tf.add(tf.matmul(x,hidden_layer_1['weights']),hidden_layer_1['bias'])
l1 = tf.nn.sigmoid(l1)

l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['bias'])
l2 = tf.nn.sigmoid(l2)

l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['bias'])
l3 = tf.nn.sigmoid(l3)

l4 = tf.add(tf.matmul(l3, hidden_layer_4['weights']), hidden_layer_4['bias'])
l4 = tf.nn.sigmoid(l4)

output = tf.matmul(l4,output_layer['weights']) + output_layer['bias']





def train_neural_network(data,testing):
    global pointer
    y_pred = output
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred ,labels=y_true))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(cross_entropy)

    #CREATE SESSION
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for step in range(4000):
            batch_x = ((X_train[pointer:pointer+100,:]).copy())
            #print(str(type(batch_x)) + "       " + str(type(x)))
            batch_y = ((y_train[pointer:pointer+100,:]).copy())
            #print(str(type(batch_y))+"       "+str(type(y_true)))
            pointer = (pointer+100)%(6800)
            sess.run(train,feed_dict={x:batch_x,y_true:batch_y})

            if(step%100==0):
                #EVALUATE MODEL
                predictions = output
                pred = tf.equal(tf.argmax(predictions,1),tf.argmax(y_true,1))
                acc = tf.reduce_mean(tf.cast(pred,tf.float32))
                print(" Accuracy  After ",step,"  Epoch " )
                print(sess.run(acc,feed_dict={x:X_test,y_true:y_test}))
                print('\n')

train_neural_network(X_train,X_test)