import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#def encode(train, test):
label_enconder = LabelEncoder().fit(train.species)
labels = label_enconder.transform(train.species)

# convert labels to onehot
cls = pd.get_dummies(labels)

#classess = list(label_enconder.classes_)

x_train = train.drop(['species', 'id'], axis = 1).values
x_test = test.drop(['id'], axis = 1).values

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#start building nn using tensorflow
x = tf.placeholder('float', [None, 192])
W = tf.Variable(tf.zeros([192, 99]))
b = tf.Variable(tf.zeros([99]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 99])

#cost function using cross-entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  sess.run(train_step, feed_dict={x: x_train, y_: cls})


#evaluate model
is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_train, y_: cls}))
