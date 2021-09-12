import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import tensorflow.compat.v1 as tf

import math
import time
from tensorflow.python.framework import graph_util
import win32api, win32con  # 终端输入命令：pip install pywin32
import easygui
import cv2
import os
tf.disable_v2_behavior()
#加载数据，包括 训练集特征与标签以及测试集特征与标签
def load_dataset():
	#读取h5文件以获取之前封存在h5文件里的数据集
	path='111'
	X=[]
	Y=[]
	i=0
	for child in os.listdir(path):
		i+=1
		child=os.path.join(path,child)
		img=cv2.imread(child)
		if img is None:
			continue
		X.append(img)
		Y.append(1)
	for child in os.listdir('222'):
		i+=1
		child=os.path.join('222',child)
		img=cv2.imread(child)
		if img is None:
			continue
		X.append(img)
		Y.append(2)
	i=0
	for child in os.listdir('333'):
		child=os.path.join('333',child)
		img=cv2.imread(child)
		i+=1
		if img is None:
			continue
		X.append(img)
		Y.append(3)
	X=np.array(X)
	Y=np.array(Y)
	# print(type(X_data))
	# 划分训练集、测试集 训练集占总数据集的90% 测试集占总数据集的10%
	# random_state=22对于数据集的拆分，它本质上也是随机的，设置不同的随机状态（或者不设置random_state参数）可以彻底改变拆分的结果。
	X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, test_size=0.1, random_state=22)
	# print(X_train.shape)
	# print(y_train[456])
	# image = Image.fromarray(X_train[456])
	# image.show()
	# y_train = y_train.reshape(1,y_train.shape[0])
	# y_test = y_test.reshape(1,y_test.shape[0])
	print(len(X_train))
	# print(X_train[0])
	#对训练集以及测试集的特征数据进行归一化处理
	X_train = X_train / 255.  # 归一化
	X_test = X_test / 255.
	# print(X_train[0])
	# one-hot
	#将标签数据一维有效化处理，这里设定的是十一位有效编码
	y_train = np_utils.to_categorical(y_train, num_classes=11)
	print(len(y_train))
	y_test = np_utils.to_categorical(y_test, num_classes=11)
	print(len(y_test))

	#返回训练集测试集的特征数据和标签一维有效编码数据
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
	#初始化一个特征-标签的容器，包含N批数据，数据是由特征与标签构成
	mini_batches = []
	np.random.seed(seed)

	#生成一个0到m的随机顺序
	permutation = list(np.random.permutation(m))
	#现在的图片顺序是按照permutation里面的索引数来排序的，以下两句是为了使图片和标签在改变顺序以后依旧对齐
	#详细可参考 https://blog.csdn.net/zhlw_199008/article/details/80569167
	shuffled_X = X[permutation]
	shuffled_Y = Y[permutation,:].reshape((m, Y.shape[1]))

	'''
		#Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
		#Partition(分割): 将已经随机化的数据集(X, Y)分割成 mini_batch_size (本文= 64)大小的子数据集。
		#尾部的数据可能小于一个mini_batch_size，所以对于最后一个mini-batch要注意处理。
	'''
	#将所有的图片数据进行分批 分为m/mini_batch_size 个批次，这里得到的是一个浮点数，也就是说可能没有被整除
	#那么对于最后一个数据集不满足mini_batch_size个来说，要区别对待
	num_complete_minibatches = math.floor(m / mini_batch_size)
	# number of mini batches of size mini_batch_size in your partitionning
	#循环 这些批次
	for k in range(0, num_complete_minibatches):
		#每次选取mini_batch_size个特征数据和mini_batch_size个标签数据放入到一个mini_batch_X和mini_batch_Y中,总共分num_complete_minibatches批次
		#比如，当k=0时，选取的就是前mini_batch_size个为一组即[0,mini_batch_size]，k=1时就从mini_batch_size开始在选取mini_batch_size个数据即[mini_batch_size,2*mini_batch_size]
		mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
		mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
		#将上面的两个一维的数组变成一个二维的mini_batch，即为一批数据（包含特征与标签）
		mini_batch = (mini_batch_X, mini_batch_Y)
		#把num_complete_minibatches批数据全部添加到特征-标签 批次中
		mini_batches.append(mini_batch)

	# Handling the end case (last mini-batch < mini_batch_size)
	#如果上面m / mini_batch_size没有被整除，则证明还有最后一批数据不满mini_batch_size个，用以下方法添加到mini_batches容器中
	if m % mini_batch_size != 0:# 若数据集和标签无法被mini_batches size整除，则把余下的数据分为一组（数据不能丢掉，浪费）
		#从num_complete_minibatches * mini_batch_size开始，一直到m 这剩下不足mini_batch_size个数据组成一个批次
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
		#同上 加入到批次容器中
		mini_batch = (mini_batch_X, mini_batch_Y)# 把mini_batch_X与mini_batch_Y组成一个元组，方便以后append进数组
		mini_batches.append(mini_batch)# 把mini_batch append进mini_batches

	#最终 返回一个有打乱顺序和带有批次的数据集 就类似于[[100,100],...,[100,100],[60,60]]这种数据集
	return mini_batches # 返回mini_batches

#训练数据
def cnn_model(X_train, y_train, X_test, y_test, keep_prob, lamda, num_epochs = 450, minibatch_size = 16):
	#批处理 输入的是一大批图片数据 即输入为[N,64,64,3]（特征） [N,11]（标签）
	X = tf.placeholder(tf.float32, [None, 64, 64, 3], name="input_x")  #输入的数据占位符
	y = tf.placeholder(tf.float32, [None, 11], name="input_y")     #输入的标签占位符
	kp = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
	lam = tf.placeholder(tf.float32, name="lamda")
	#两层卷积层、两层池化层、一层全连接层，一层softmax输出
	#conv1
	#设定卷积层一：卷积核的大小为5*5，初试输入的图片深度为3（通道数，RGB图片），设定32个滤镜，产生32张图片
	#然后使用relu激活函数 即从N*64*64*3到N*64*64*32
	W_conv1 = weight_variable([5,5,3,32])
	b_conv1 = bias_variable([32])
	z1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
	#使用一个2*2的最大化池化层 将图片大小缩小一半，变成[N,32,32,32]
	maxpool1 = max_pool_2x2(z1) #max_pool1完后maxpool1维度为[?,32,32,32]

	# 设定卷积层一：卷积核的大小为5*5，初试输入的图片深度为32（上一层产生了32张图片），设定64个滤镜，产生64张图片
	# 然后使用relu激活函数 即从N*64*64*32到N*64*64*64
	W_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])
	z2 = tf.nn.relu(conv2d(maxpool1, W_conv2) + b_conv2)
	# 使用一个2*2的最大化池化层 将图片大小缩小一半，变成[N,16,16,64]
	maxpool2 = max_pool_2x2(z2) #max_pool2,shape [?,16,16,64]

	#conv3  效果比较好的一次模型是没有这一层，只有两次卷积层，隐藏单元100，训练20次
	# W_conv3 = weight_variable([5, 5, 64, 128])
	# b_conv3 = bias_variable([128])
	# z3 = tf.nn.relu(conv2d(maxpool2, W_conv3) + b_conv3)
	# maxpool3 = max_pool_2x2(z3)  # max_pool3,shape [?,8,8,128]

	#full connection1
	#全连接层隐藏层，设定下一层的隐藏层神经元个数为200
	W_fc1 = weight_variable([16*16*64, 200])
	b_fc1 = bias_variable([200])
	#将上面通过卷积池化得到的图片变成一维的数据即16*16*64的一维数组，输入到平坦层
	maxpool2_flat = tf.reshape(maxpool2, [-1, 16*16*64])
	#平坦层与第一层隐藏层全连接并使用relu激活函数
	z_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, W_fc1) + b_fc1)
	#在这一步 即平坦层到第一层隐藏层随机保留多少个神经元，默认是1（全部） 传入的参数是0.8，即舍弃20%保留80%
	z_fc1_drop = tf.nn.dropout(z_fc1, keep_prob=kp)

	#softmax layer
	# 输出层，设定下一层的输出层神经元个数为11 即代表上面我们将标签一维有效化11位的数据
	#上一层的隐藏层为200个神经元，这里输出为11个神经元
	W_fc2 = weight_variable([200, 11])
	b_fc2 = bias_variable([11])
	z_fc2 = tf.add(tf.matmul(z_fc1_drop, W_fc2),b_fc2, name="outlayer")
	#最后一层输出层使用的激活函数为softmax 得到一个概率输出，最终会化成一个11位的有效编码
	prob = tf.nn.softmax(z_fc2, name="probability")

	#cost function   规则化
	#设定一个正则化规则 lam参数表示了正则化项的权重，也就是公式J（θ）＋λR(w）中的λ。w为需要计算正则化损失的参数
	# regularizer = tf.contrib.layers.l2_regularizer(lam)
	# #计算得到一个正则化的值
	# regularization = regularizer(W_fc1) + regularizer(W_fc2)
	regularzation_rate=0.5
	regularization = regularzation_rate * tf.nn.l2_loss(W_fc1) + regularzation_rate * tf.nn.l2_loss(W_fc2)
	#tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z_fc2)函数用来设定损失函数
	#tf.reduce_mean求内部参数的均值
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z_fc2)) + regularization

	#设定一个 优化器 cost即最小化目标变量，一般就是训练的目标函数，交叉熵
	train = tf.train.AdamOptimizer().minimize(cost)
	# output_type='int32', name="predict"
	#tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引 0表示列，1表示行，返回他们的索引值
	#pred是指得到的神经网络输出每一行最大值的索引值的数组
	pred = tf.argmax(prob, 1, output_type="int32", name="predict")  # 输出结点名称predict方便后面保存为pb文件

	#神经网络处理完的数据与真实标签数据对比 并返回一个bool类型的张量
	correct_prediction = tf.equal(pred, tf.argmax(y, 1, output_type='int32'))
	#将bool类型的序列转化为0和1的序列 tf.reduce_mean计算这个序列的均值
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#随机生成种子
	tf.set_random_seed(1)  # to keep consistent results

	seed = 0

	#初始化神经网络结构
	init = tf.global_variables_initializer()
	#开启一个会话层
	with tf.Session() as sess:
		#初始化
		sess.run(init)
		#训练次数num_epochs 即循环训练num_epochs次
		for epoch in range(num_epochs):
			seed = seed + 1
			#初始化损失值为0
			epoch_cost = 0.
			#分批，从h5取出来的训练集特征数据除以minibatch_size（默认16，传进来的是16）=批次
			num_minibatches = int(X_train.shape[0] / minibatch_size)
			#获取上面获得的批次数据 传入总的特征数据集，标签数据集，还有每个批次的数据量大小以及随机种子
			minibatches = random_mini_batches(X_train, y_train, minibatch_size, seed)
			#通过一个循环来获取每一个批次的数据
			for minibatch in minibatches:
				#获取每一批次的特征与标签数据
				(minibatch_X, minibatch_Y) = minibatch
				#在run里添加之前定义好的优化器，损失函数，然后喂数据 数据就是每一批的特征与标签，还有就是lam参数表示了正则化项的权重以及kb参数（保留的神经元比例）
				#minibatch_cost是sess.run每一次循环当前返回的损失值
				_, minibatch_cost = sess.run([train, cost], feed_dict={X: minibatch_X, y: minibatch_Y, kp: keep_prob, lam: lamda})
				#epoch_cost循环了N次的损失值
				epoch_cost += minibatch_cost / num_minibatches
			#每训练10次打印一次损失值和当前时间
			if epoch % 10 == 0:
				#Cost after epoch 0: 1.536390
				#2021-04-22 15:59:07

				print("Cost after epoch %i: %f" % (epoch, epoch_cost))
				print(str((time.strftime('%Y-%m-%d %H:%M:%S'))))

		# 这个accuracy是前面的accuracy，tensor.eval()和Session.run区别很小
		#这里train_acc用来测试 训练的准确率大小 取得是训练集最后1000个数据（上面训练时数据顺序是打乱的，这里没有）
		train_acc = accuracy.eval(feed_dict={X: X_train[:1000], y: y_train[:1000], kp: 0.8, lam: lamda})
		#打印训练准确率
		print("train accuracy", train_acc)
		# 这里test_acc用来测试模型的的准确率大小 取得是测试集集最后1000个数据，即没有被训练过得数据来测试我的模型准确率
		test_acc = accuracy.eval(feed_dict={X: X_test[:1000], y: y_test[:1000], lam: lamda})
		#打印准确率 这个准确率是应用准确率，更靠谱
		print("test accuracy", test_acc)

		#save model
		#以下是用来保存模型的设定 下面是要保存模型的各种结构
		saver = tf.train.Saver({'W_conv1':W_conv1, 'b_conv1':b_conv1, 'W_conv2':W_conv2, 'b_conv2':b_conv2,
		                        'W_fc1':W_fc1, 'b_fc1':b_fc1, 'W_fc2':W_fc2, 'b_fc2':b_fc2})
		#保存模型为model_500_200_c2//cnn_model.ckpt
		#global_step=1000 设置多少步一保存，但是添加这个步数就要把模型保存函数写到循环内
		#训练完所有数据以后 保存模型
		saver.save(sess, "model_500_200_c2//cnn_model.ckpt")

		'''
			还有一种保存模型的方法，只保存模型的参数，不保存结构，所以使用得时候需要自行定义结构，但是这种方法更省空间
			即存储时：torch.save(模型名称.state_dict,'model.pkl') 
			加载时： 模型名称.load_state_dict(torch.load('model.pkl'))
		'''




		#将训练好的模型保存为.pb文件，方便在Android studio中使用
		# output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['predict'])
		# with tf.gfile.FastGFile('model_500_200_c2//digital_gesture.pb', mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
		# 	f.write(output_graph_def.SerializeToString())

def start_train():
	title = "训练样本集"
	msg = "请输入你希望训练集被训练算法遍历的次数epoch"
	#获取用户输入的训练次数
	epoch = int(easygui.enterbox(msg,title))
	time1 = str(time.strftime('%Y-%m-%d %H:%M:%S'))
	win32api.MessageBox(0,"载入数据集！%s" %time1,"提醒",win32con.MB_OK)
	# print("载入数据集: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
	#获取数据
	X_train, X_test, y_train, y_test = load_dataset()
	time2 = str(time.strftime('%Y-%m-%d %H:%M:%S'))
	win32api.MessageBox(0, "开始训练:%s，请稍后..." % time2, "提醒", win32con.MB_OK)
	# print("开始训练: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
	#将数据送到神经网络中进行训练
	cnn_model(X_train, y_train, X_test, y_test, 0.8, 1e-4, num_epochs=epoch, minibatch_size=16)
	# print("训练结束: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
	time2 = str(time.strftime('%Y-%m-%d %H:%M:%S'))
	win32api.MessageBox(0, "训练结束！%s" % time2, "提醒", win32con.MB_OK)


if __name__ == "__main__":
	start_train()

