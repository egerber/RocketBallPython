import tensorflow as tf



data=tf.placeholder(tf.float32,[2,4])
target=tf.placeholder(tf.float32,[2,4])
#rmse=tf.sqrt(tf.square(tf.subtract(target,data)),name="RMSE")
slice1=tf.slice(target,[0,0],[2,2])
slice2=tf.slice(data,[0,2],[2,2])
data2=tf.concat([slice1,slice2],1)
sess=tf.Session()
res=sess.run([slice1,slice2,data2],
             feed_dict={data:[[0.2,0.3,0.4,0.5],[-0.2,-0.3,-0.4,-0.5]],
                        target:[[0.3,0.7,0.4,0.6],[-0.3,-0.7,-0.4,-0.6]]
                        }
             )
print(res)