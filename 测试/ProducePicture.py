import os
from PIL import Image
import numpy as np
import h5py
import tensorflow as tf
import imageio
from skimage import img_as_ubyte
import cv2
import easygui
import win32api, win32con
import datetime

def takePhoto():
    title = "采集手势样本"
    msg = '请输入手势编码'
    name = easygui.enterbox(msg, title)

    win32api.MessageBox(0, "请按's'键保存手势样本！按'q'键退出采集！注意手势置于框中", "提醒", win32con.MB_OK)

    savedpath = r'./img/produce'
    isExists = os.path.exists(savedpath)
    if not isExists:
        os.makedirs(savedpath)
        print('path of %s is build' % (savedpath))
    else:
        print('path of %s already exist and rebuild' % (savedpath))

    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 初始图像是RGB格式，转换成BGR即可正常显示了
        frame = cv2.flip(frame, 2)
        cv2.rectangle(frame, (150, 150), (450, 450), (0, 255, 0))  # 画出截取的手势框图

        cv2.imshow("test", frame)
        now = datetime.datetime.now()
        now = now.strftime('%m-%d-%H-%M-%S')
        savedname = '/' + name + '_test.jpg'.format(now)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # s按N拍摄
            cv2.imwrite(savedpath + savedname, frame)
            cv2.namedWindow("Image")
            cv2.imshow("Image", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def getHand():
    # 指明被遍历的文件夹
    rootdir = r'./img/produce'
    for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
        for filename in filenames:
            if os.path.splitext(filename)[1]==".jpg":
                print(os.path.splitext(filename)[0],',',os.path.splitext(filename)[1])
                print('parent is :' + parent)
                print('filename is :' + filename)
                currentPath = os.path.join(parent, filename)
                print('the fulll name of the file is :' + currentPath)

                img = Image.open(currentPath)
                print(img.format, img.size, img.mode)

                box1 = (150,150,450, 450)  # 设置左、上、右、下的像素
                image1 = img.crop(box1)  # 图像裁剪
                image1.save(r"./img/produce/" + filename)  # 存储裁剪得到的图像

#压缩图片,把图片压缩成64*64的
'''
    tensorflow里面给出了一个函数用来读取图像，不过得到的结果是最原始的图像，是没有经过解码的图像，
    这个函数为tf.gfile.FastGFile（‘path’， ‘r’）.read()。
    如果要显示读入的图像，那就需要经过解码过程，tensorflow里面提供解码的函数有两个，
    tf.image.decode_jepg和tf.image.decode_png分别用于解码jpg格式和png格式的图像进行解码，
    得到图像的像素值，这个像素值可以用于显示图像。如果没有解码，读取的图像是一个字符串，没法显示。
'''
def resize_img():
	dirs = os.listdir("./img/produce")
	for filename in dirs:
		if not filename.endswith(".jpg"):
			continue
        #获取文本操作句柄，类似于python提供的文本操作open()函数，filename是要打开的文件名，mode是以何种方式去读写，将会返回一个文本操作句柄。
		im = tf.gfile.GFile("./img/produce//{}".format(filename), 'rb').read()
		# print("正在处理第%d张照片"%counter)
		with tf.compat.v1.Session() as sess:
            #对上面通过TensorFlow操作的图片进行解码
			img_data = tf.image.decode_jpeg(im)
            #这里tf.image.decode_jpeg 得到的是uint8格式，范围在0-255之间，经过convert_image_dtype 就会被转换为区间在0-1之间的float32格式，
			image_float = tf.image.convert_image_dtype(img_data, tf.float32)
            #通过tf.image.resize_images函数调整图像的大小。这个函数第一个参数为原始图像，第二个和第三个分别为调整后图像的大小，method参数给出了调整图像大小的算法
            #method = 0 双线性插值法method = 1 最近邻居法method = 2 双三次插值法method = 3 面积插值法
			resized = tf.image.resize(image_float, [64, 64], method=3)
            #eval()也是启动计算的一种方式。基于Tensorflow的基本原理，首先需要定义图，然后计算图，其中计算图的函数常见的有run()函数，如sess.run()。同样eval()也是此类函数，
			resized_im = resized.eval()

			imageio.imwrite("./img/produce//{}".format(filename),img_as_ubyte(resized_im))


#图片转h5文件
def image_to_h5():
    #遍历./img/train_img_all所有图片
	dirs = os.listdir("./img/produce")
	Y = [] #label标签
	X = [] #data数据
	# print(len(dirs))
	for filename in dirs:
		if not filename.endswith(".jpg"):
			continue
        #将1-1,2-1这种切割成1,2这种，代表的是手势编码，也就是标签
		label = int(filename.split('_')[0])
		# print(label)
        #添加到Y数组里
		Y.append(label)
        #打开图片
		im = Image.open("./img/produce//{}".format(filename)).convert('RGB')
        #将图片转换为矩阵数组，即训练或者测试用的特征数据
		mat = np.asarray(im) #image 转矩阵
        #添加到X数组里
		X.append(mat)

    #创建一个h5的文件，这个文件的作用是成为后续训练或者测试的数据集
	file = h5py.File("./img/produce//produce.h5","w")
    #分别将特征数据X与该数据代表的含义即标签 转为numpy数组保存到h5文件里
	file.create_dataset('X', data=np.array(X))
	file.create_dataset('Y', data=np.array(Y))
    #关闭流文件
	file.close()

def produce():
    takePhoto()
    getHand()
    resize_img()
    image_to_h5()

if __name__ == "__main__":
    produce()



