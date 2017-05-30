import tensorflow as tf


sess=tf.Session()
print(sess.run(tf.one_hot(tf.argmax(tf.constant([[[0.9,0.1,0.04,0.4],[0.2,0.5,0.1,0.9]]]),2),4)))