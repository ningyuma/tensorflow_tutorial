import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split


# read the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# define encoder function
def encode(train, test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)  # encode species strings  **to numerical
    classes = list(le.classes_)  # save column names for submission  **like unique
    test_ids = test.id  # save test ids for submission
    x_train = train.drop(['species', 'id'], axis=1).values  # drop off col
    x_test = test.drop(['id'], axis=1).values

    return x_train, labels, x_test, test_ids, classes

x_train, labels, x_test, test_ids, classes = encode(train, test)

# convert labels to onehot
cls = pd.get_dummies(labels)

# standardize data to mean = 0, std = 1
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train, x_val, y_train, y_val = train_test_split(x_train, cls, test_size=0.33, random_state=42)
# print(x_train.shape)
# print(y_train.shape)

x = tf.placeholder(tf.float32, [None, 192])
W = tf.Variable(tf.zeros([192, 99]))
b = tf.Variable(tf.zeros([99]))

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 99])  # input the correct answers
# tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1]
# tf.reduce_mean computes the mean over all the examples in the batch.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
x_train_df = pd.DataFrame(x_train)
# for _ in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
for _ in range(1000):
    # batch = x_train_df.next_batch(100)
    train_step.run(feed_dict={x: x_train, y_: y_train})
# tf.argmax(y,1) is the label our model thinks is most likely for each input,
# while tf.argmax(y_,1) is the correct label.
# evaluate model
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_prediction = tf.argmax(y, 1)
# To determine what fraction are correct,
# we cast to floating point numbers and then take the mean.
# For example, [True, False, True, True]
# would become [1,0,1,1] which would become 0.75.

# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: x_test}))


test_ids = test.pop('id')
submission = pd.DataFrame(correct_prediction, index=test_ids, columns=classes)
submission.to_csv('submission_tf.csv')
