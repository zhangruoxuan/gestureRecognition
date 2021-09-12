import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import time
from tensorflow.python.framework import graph_util
import win32api, win32con  # 终端输入命令：pip install
import easygui


#load dataset
def load_dataset():
	#划分训练集、测试集
	data = h5py.File("dataset//data.h5","r")
	X_data = np.array(data['X']) #data['X']是h5py._hl.dataset.Dataset类型，转化为array
	Y_data = np.array(data['Y'])
	# print(type(X_data))
	X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, train_size=0.9, test_size=0.1, random_state=22)
	# print(X_train.shape)
	# print(y_train[456])
	# image = Image.fromarray(X_train[456])
	# image.show()
	# y_train = y_train.reshape(1,y_train.shape[0])
	# y_test = y_test.reshape(1,y_test.shape[0])
	print(X_train.shape)
	# print(X_train[0])
	X_train = X_train / 255.  # 归一化
	X_test = X_test / 255.
	# print(X_train[0])
	# one-hot
	y_train = np_utils.to_categorical(y_train, num_classes=11)
	print(y_train.shape)
	y_test = np_utils.to_categorical(y_test, num_classes=11)
	print(y_test.shape)

	return X_train, X_test, y_train, y_test

#定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
	tf.set_random_seed(1)
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

#定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
	return tf.Variable(tf.constant(0.0, shape=shape))

#定义一个函数，用于构建卷积层
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#定义一个函数，用于构建池化层
def max_pool_2x2(z):
	return tf.nn.max_pool(z, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#
def random_mini_batches(X, Y, mini_batch_size=16, seed=0):#传入需要分组的数据集和与之相应的标签，并指定mini_batches的大小
	"""
	Creates a list of random minibatches from (X, Y)
	            加快神经网络学习速度的优化方法：Mini-Batch梯度下降
	            将整个训练集划分成若干个小的训练集来依次训练
	Arguments:
	X -- input data, of shape (input size, number of examples)
	Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
	mini_batch_size - size of the mini-batches, integer
	seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

	Returns:
	mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
	"""

	m = X.shape[0]  # # 获取训练集中图片的个数
	mini_batches = []
	np.random.seed(seed)

	# Step 1: Shuffle (X, Y)# 进行随机操作
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation]
	shuffled_Y = Y[permutation,:].reshape((m, Y.shape[1]))

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	#Partition(分割): 将已经随机化的数据集(X, Y)分割成 mini_batch_size (本文= 64)大小的子数据集。
	# 尾部的数据可能小于一个mini_batch_size，所以对于最后一个mini-batch要注意处理。
	num_complete_minibatches = math.floor(m / mini_batch_size)
	# number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
		mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:# 若数据集和标签无法被mini_batches size整除，则把余下的数据分为一组（数据不能丢掉，浪费）
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]

		mini_batch = (mini_batch_X, mini_batch_Y)# 把mini_batch_X与mini_batch_Y组成一个元组，方便以后append进数组
		mini_batches.append(mini_batch)# 把mini_batch append进mini_batches

	return mini_batches # 返回mini_batches


def cnn_model(X_train, y_train, X_test, y_test, keep_prob, lamda, num_epochs = 450, minibatch_size = 16):
	X = tf.placeholder(tf.float32, [None, 64, 64, 3], name="input_x")  #输入的数据占位符
	y = tf.placeholder(tf.float32, [None, 11], name="input_y")     #输入的标签占位符
	kp = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
	lam = tf.placeholder(tf.float32, name="lamda")
	#两层卷积层、两层池化层、一层全连接层，一层softmax输出
	#conv1
	W_conv1 = weight_variable([5,5,3,32])
	b_conv1 = bias_variable([32])
	z1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
	maxpool1 = max_pool_2x2(z1) #max_pool1完后maxpool1维度为[?,32,32,32]

	#conv2
	W_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])
	z2 = tf.nn.relu(conv2d(maxpool1, W_conv2) + b_conv2)
	maxpool2 = max_pool_2x2(z2) #max_pool2,shape [?,16,16,64]

	#conv3  效果比较好的一次模型是没有这一层，只有两次卷积层，隐藏单元100，训练20次
	# W_conv3 = weight_variable([5, 5, 64, 128])
	# b_conv3 = bias_variable([128])
	# z3 = tf.nn.relu(conv2d(maxpool2, W_conv3) + b_conv3)
	# maxpool3 = max_pool_2x2(z3)  # max_pool3,shape [?,8,8,128]

	#full connection1
	W_fc1 = weight_variable([16*16*64, 200])
	b_fc1 = bias_variable([200])
	maxpool2_flat = tf.reshape(maxpool2, [-1, 16*16*64])
	z_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, W_fc1) + b_fc1)
	z_fc1_drop = tf.nn.dropout(z_fc1, keep_prob=kp)

	#softmax layer
	W_fc2 = weight_variable([200, 11])
	b_fc2 = bias_variable([11])
	z_fc2 = tf.add(tf.matmul(z_fc1_drop, W_fc2),b_fc2, name="outlayer")
	prob = tf.nn.softmax(z_fc2, name="probability")

	#cost function   规则化
	#regularizer = tf.nn.l2_loss(lam)
	regularization = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z_fc2)) + regularization

	train = tf.train.AdamOptimizer().minimize(cost)
	# output_type='int32', name="predict"
	pred = tf.argmax(prob, 1, output_type="int32", name="predict")  # 输出结点名称predict方便后面保存为pb文件
	correct_prediction = tf.equal(pred, tf.argmax(y, 1, output_type='int32'))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.set_random_seed(1)  # to keep consistent results

	seed = 0

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			seed = seed + 1
			epoch_cost = 0.
			num_minibatches = int(X_train.shape[0] / minibatch_size)
			minibatches = random_mini_batches(X_train, y_train, minibatch_size, seed)
			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch
				_, minibatch_cost = sess.run([train, cost], feed_dict={X: minibatch_X, y: minibatch_Y, kp: keep_prob, lam: lamda})
				epoch_cost += minibatch_cost / num_minibatches
			if epoch % 10 == 0:
				print("Cost after epoch %i: %f" % (epoch, epoch_cost))
				print(str((time.strftime('%Y-%m-%d %H:%M:%S'))))

		# 这个accuracy是前面的accuracy，tensor.eval()和Session.run区别很小
		train_acc = accuracy.eval(feed_dict={X: X_train[:1000], y: y_train[:1000], kp: 0.8, lam: lamda})
		print("train accuracy", train_acc)
		test_acc = accuracy.eval(feed_dict={X: X_test[:1000], y: y_test[:1000], lam: lamda})
		print("test accuracy", test_acc)

		#save model
		saver = tf.train.Saver({'W_conv1':W_conv1, 'b_conv1':b_conv1, 'W_conv2':W_conv2, 'b_conv2':b_conv2,
		                        'W_fc1':W_fc1, 'b_fc1':b_fc1, 'W_fc2':W_fc2, 'b_fc2':b_fc2})
		saver.save(sess, "model_500_200_c2//cnn_model.ckpt")
		#将训练好的模型保存为.pb文件，方便在Android studio中使用
		# output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['predict'])
		# with tf.gfile.FastGFile('model_500_200_c2//digital_gesture.pb', mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
		# 	f.write(output_graph_def.SerializeToString())

def start_train():
	title = "训练样本集"
	msg = "请输入你希望训练集被训练算法遍历的次数epoch"
	epoch = int(easygui.enterbox(msg,title))
	time1 = str(time.strftime('%Y-%m-%d %H:%M:%S'))
	win32api.MessageBox(0,"载入数据集！%s" %time1,"提醒",win32con.MB_OK)
	# print("载入数据集: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
	X_train, X_test, y_train, y_test = load_dataset()
	time2 = str(time.strftime('%Y-%m-%d %H:%M:%S'))
	win32api.MessageBox(0, "开始训练:%s，请稍后..." % time2, "提醒", win32con.MB_OK)
	# print("开始训练: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
	cnn_model(X_train, y_train, X_test, y_test, 0.8, 1e-4, num_epochs=epoch, minibatch_size=16)
	# print("训练结束: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
	time2 = str(time.strftime('%Y-%m-%d %H:%M:%S'))
	win32api.MessageBox(0, "训练结束！%s" % time2, "提醒", win32con.MB_OK)


if __name__ == "__main__":
	start_train()