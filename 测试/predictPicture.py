import tensorflow.compat.v1 as tf

import cnn
import numpy as np
import h5py
from keras.utils import np_utils
import predictVedio as pre
import cv2
import os

#加载之前训练好的模型 即CNN文件里的模型结构加载
def load_parameters():
	#tf.reset_default_graph函数用于清除默认图形堆栈并重置全局默认图形 简单理解就是初始化
	tf.reset_default_graph()
	#以下是创建新的tensorflow变量 名称name、变量规格shape 分别对应cnn文件里的模型结构
	W_conv1 = tf.get_variable("W_conv1",shape = [5,5,3,32])
	b_conv1 = tf.get_variable("b_conv1", shape = [32])
	W_conv2= tf.get_variable("W_conv2", shape=[5, 5, 32, 64])
	b_conv2 = tf.get_variable("b_conv2", shape=[64])
	W_fc1 = tf.get_variable("W_fc1", shape = [16*16*64, 200])
	b_fc1 = tf.get_variable("b_fc1", shape = [200])
	W_fc2 = tf.get_variable("W_fc2", shape=[200, 11])
	b_fc2 = tf.get_variable("b_fc2", shape=[11])

	#模型结构化 类似于声明一个模型类
	parameters = {}
	#模型的保存和加载  这里用来加载模型
	saver = tf.train.Saver()
	with tf.Session() as sess:
		#下面的restore就是在当前的sess下恢复了所有的变量
		saver.restore(sess, "model_500_200_c2//./cnn_model.ckpt")
		# print(W_conv1.eval())
		#相当于把恢复后的卷积层，池化层，全连接层激活。并放进parameters中 此时parameters相当于一个模型类
		#这个类里包含了模型各种结构和激活函数
		parameters["W_conv1"] = W_conv1.eval()
		parameters["b_conv1"] = b_conv1.eval()
		parameters["W_conv2"] = W_conv2.eval()
		parameters["b_conv2"] = b_conv2.eval()
		parameters["W_fc1"] = W_fc1.eval()
		parameters["b_fc1"] = b_fc1.eval()
		parameters["W_fc2"] = W_fc2.eval()
		parameters["b_fc2"] = b_fc2.eval()

	#返回的就是个模型
	return parameters

#此函数用来预测新的数据的结果 传入模型和图片特征 这个x在本函数里设定的是一次传进来一个
def predict(parameters, X):
	#获得parameters中 模型的各种结构
	W_conv1 = parameters["W_conv1"]
	b_conv1 = parameters["b_conv1"]
	W_conv2 = parameters["W_conv2"]
	b_conv2 = parameters["b_conv2"]
	W_fc1 = parameters["W_fc1"]
	b_fc1 = parameters["b_fc1"]
	W_fc2 = parameters["W_fc2"]
	b_fc2 = parameters["b_fc2"]

	#这里N代表每次喂多少个数据 如果设置为1  则每次只能识别1个 这里我设置为测试集的大小N 即可以一次性识别N张图片
	N=X.shape[0]
	#print('x[0]',X.shape[0])
	#输入的数据为N张图片的特征数据
	x = tf.placeholder(tf.float32, [N, 64, 64, 3])
	#以下就是卷积池化的效果 两层卷积两层池化
	z1 = tf.nn.relu(cnn.conv2d(x, W_conv1) + b_conv1)
	maxpool1 = cnn.max_pool_2x2(z1)
	z2 = tf.nn.relu(cnn.conv2d(maxpool1, W_conv2) + b_conv2)
	maxpool2 = cnn.max_pool_2x2(z2)
	#这个是平坦层 卷积池化以后获得了64张图片 图片的大小为16*16 所以平坦层有16*16*64个神经元作为输入
	maxpool2_flat = tf.reshape(maxpool2, [-1, 16 * 16 * 64])
	#这是隐藏层
	z_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, W_fc1) + b_fc1)
	#这是输出层 #得到一个1*11的数组 1为图片个数 11为对应标签(一位有效编码为11位)
	logits = tf.nn.softmax(tf.matmul(z_fc1, W_fc2) + b_fc2)
	#按照每行元素最大值的 列索引，即每张图片所对应的一位有效编码那个1所在的位置 即得到输出结果
	c = tf.argmax(logits, 1)

	#打开会话层 启动计算图
	with tf.Session() as sess:
		#得到这11位编码的概率 以及最大概率的索引
		prediction, logit = sess.run([c,logits], feed_dict={x: X})
	#设置打印数据格式
	np.set_printoptions(suppress=True)
	#打印概率和索引
	#print(logit)
	print('here!',prediction)
	#返回索引值 0代表的是第一行最大列索引值 即第一张图片的识别结果
	#result=prediction[0]
	return prediction

def loadData(url):
	# 加载数据，包括特征与标签
	# 读取h5文件以获取之前封存在h5文件里的数据集
	data = h5py.File(url, "r")
	# 获取特征值
	X_data = np.array(data['X'])  # data['X']是h5py._hl.dataset.Dataset类型，转化为array
	# 获取标签值
	Y_data = np.array(data['Y'])
	#print(X_data.shape)
	# 对训练集以及测试集的特征数据进行归一化处理
	X_train = X_data / 255.  # 归一化
	# 将标签数据一维有效化处理，这里设定的是十一位有效编码
	y_train = np_utils.to_categorical(Y_data, num_classes=11)
	#print(y_train.shape)

	#返回h5文件的特征数据以及标签数据
	return X_train, y_train

def recognitionPicture():
	# 我们用来测试的数据的h5文件的地址 测试数据一般用一张图片的h5文件
	url = "./img/vedioout//vedioout.h5"
	# 加载一个模型 即 继承我们之前训练的那个模型结构，然后用来预测新的数据
	parameters = load_parameters()
	# 通过读入h5文件 获得图片的特征数据X以及标签数据y，此时用来预测没有用到y
	dirs = os.listdir("./img/resize")
	Y = []  # label标签
	X = []  # data数据
	# print(len(dirs))
	for filename in dirs:
		if not filename.endswith(".jpg"):
			continue
	X.append(cv2.imread("./img/resize//{}".format(filename)))
	# 预测新的数据
	result = predict(parameters, X)
	parameters.clear()

	# 以下是得到的序列转为所对应的文字信息
	# print('result的类型',type(result))
	# 将numpy.ndarray转为list 上面通过识别以后为numpy.ndarray格式的数据，要想去邻重 需要转为list格式
	result = result.tolist()
	# print('result的类型',type(result))
	# 去邻重复
	result = pre.del_adjacent(result)
	print('未转化为文字的标签序列为：', result)
	# 以下方法是将序列转为文字描述
	# 先读取excel表中的数据
	cols = pre.read_excel()
	# 定义一个 文字描述的数组
	finalResult = []
	# index是下标索引 item是值
	for index, item in enumerate(result):
		# 将序列转为文字，并存入数组中
		print(index)
		msg = cols[int(result[index])]  # 或者这样写 msg = cols[int(item)]
		# print(msg)
		finalResult.append(msg)
	# print(finalResult)
	# 这个是数组里所以元素合并在一块儿的字符串
	addResult = ''
	for value in finalResult:
		addResult += value

	return finalResult, addResult

# print(addResult)

if __name__ == '__main__':
	finalResult, addResult = recognitionPicture()

	print('识别的序列为：', finalResult)
	print('识别的结果为：', addResult)