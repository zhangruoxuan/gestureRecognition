import os
from PIL import Image
import numpy as np
import h5py
import tensorflow as tf
import imageio
from skimage import img_as_ubyte
import cv2
import random
import easygui
import win32api, win32con
import datetime

def takePhoto():
    title = "采集手势样本"
    msg = '请输入手势编码'
    name = easygui.enterbox(msg, title)

    win32api.MessageBox(0, "请按's'键保存手势样本！按'q'键退出采集！注意手势置于框中", "提醒", win32con.MB_OK)

    savedpath = r'./img/testlixin'
    isExists = os.path.exists(savedpath)
    if not isExists:
        os.makedirs(savedpath)
        print('path of %s is build' % (savedpath))
    else:
        print('path of %s already exist and rebuild' % (savedpath))
    i = get_num1(savedpath,name)

    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 初始图像是RGB格式，转换成BGR即可正常显示了
        frame = cv2.flip(frame, 2)
        cv2.rectangle(frame, (150, 150), (450, 450), (0, 255, 0))  # 画出截取的手势框图

        cv2.imshow("test", frame)
        now = datetime.datetime.now()
        now = now.strftime('%m-%d-%H-%M-%S')
        savedname = '/' + name + '_' + str(i+1) + '.jpg'.format(now)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # s按N拍摄
            i += 1
            cv2.imwrite(savedpath + savedname, frame)
            cv2.namedWindow("Image")
            cv2.imshow("Image", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    num_later = get_num1(savedpath,name)
    win32api.MessageBox(0, "已录入手势编码'%s'共%s张手势样本!请点击'处理手势样本'按钮处理样本！" %(name,num_later), "提醒", win32con.MB_OK)

def get_num1(path,name):
    #返回一个文件夹的文件目录列表
    files = os.listdir(path)
    #print(files)
    res = [0] * 100
    for f in files:
        #print(f,',',f[0],',',f[1],',',f[2],',',f[3],',',f[4],',',f[5])
        if not f.endswith(".jpg"):
            continue
        else:
            #print(f[0],',',int(f[0]))
            #f[0]即是1-1.jpg中的1 2-1.jpg中的2等
            res[int(f[0])] += 1
    # print(res[int(name)])
    return res[int(name)]


def getHand():
    # 指明被遍历的文件夹
    rootdir = r'./img/testlixin'
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
                image1.save(r"./img/testlixin2/" + filename)  # 存储裁剪得到的图像

#压缩图片,把图片压缩成64*64的
'''
    tensorflow里面给出了一个函数用来读取图像，不过得到的结果是最原始的图像，是没有经过解码的图像，
    这个函数为tf.gfile.FastGFile（‘path’， ‘r’）.read()。
    如果要显示读入的图像，那就需要经过解码过程，tensorflow里面提供解码的函数有两个，
    tf.image.decode_jepg和tf.image.decode_png分别用于解码jpg格式和png格式的图像进行解码，
    得到图像的像素值，这个像素值可以用于显示图像。如果没有解码，读取的图像是一个字符串，没法显示。
'''
def resize_img():
	dirs = os.listdir("./img/testlixin2")
	for filename in dirs:
		if not filename.endswith(".jpg"):
			continue
        #获取文本操作句柄，类似于python提供的文本操作open()函数，filename是要打开的文件名，mode是以何种方式去读写，将会返回一个文本操作句柄。
		im = tf.gfile.GFile("./img/testlixin2//{}".format(filename), 'rb').read()
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

			imageio.imwrite("./img/testlixin3//{}".format(filename),img_as_ubyte(resized_im))

def rotate_main():
    path = './img/testlixin3'
    path_all = './img/testlixin4'
    res,num_lei = get_num(path)
    #在种类间循环例如1-2,2-2,3-2个种类
    for i in range(1,num_lei + 1):
        #获取每个种类比如1或者2或者3的图片的个数
        num = res[i]
        #在这么些张图片进行遍历
        for j in range(1,res[i]+1):
            #将这些图片的路径提取出来
            roi = cv2.imread(path + '/' + str(i)+ '_' + str(j) + '.jpg')
            #重新写到path_all下
            cv2.imwrite(path_all + '/' + str(i) + '_' + str(j) + '.jpg',roi)
            #循环5次
            for k in range(5):
                #对图像旋转
                img_rotation = rotate(roi)  # 旋转
                #图片数目加1，这里是为了后续命名跟在之前的后面，比如之前有1-32 现在翻转以后命名为1-33
                num = num + 1
                #将翻转的数据写到该文件夹下
                cv2.imwrite(path_all + '/' + str(i) + '_' + str(num) + '.jpg', img_rotation)
                #再对上面旋转后的图像进行翻转 第二个参数>0: 沿y-轴翻转, 0: 沿x-轴翻转, <0: x、y轴同时翻转
                img_flip = cv2.flip(img_rotation, 1)  # 翻转
                #再次加1
                num = num + 1
                #写入进去
                cv2.imwrite(path_all + '/' + str(i) + '_' + str(num) + '.jpg', img_flip)
        print("%s完成" % i)

#用来获取目录里手势种类的数目以及该种类有多少图片集
def get_num(path):
    files = os.listdir(path)
    res = [0] * 100
    num_lei = 0
    for f in files:
        if not f.endswith(".jpg"):
            continue
        else:
            res[int(f[0])] += 1
    for i in res:
            if i !=0:
                num_lei += 1
    return res,num_lei

#随机旋转图片并获取新的图片
def rotate(image, scale=0.9):
    # 随机角度
    angle = random.randrange(-90, 90)
    #获取长和宽
    w = image.shape[1]
    h = image.shape[0]
    #getRotationMatrix2D主要用于获得图像绕着 某一点的旋转矩阵
    #(w/2,h/2)表示旋转的中心点，angle表示表示旋转的角度，scale表示图像缩放因子
    #M是获取的旋转矩阵
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #根据旋转矩阵进行仿射变换
    image = cv2.warpAffine(image,M,(w,h))
    return image


#图片转h5文件
def image_to_h5():
    #遍历./img/train_img_all所有图片
	dirs = os.listdir("./img/testlixin4")
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
		im = Image.open("./img/testlixin4//{}".format(filename)).convert('RGB')
        #将图片转换为矩阵数组，即训练或者测试用的特征数据
		mat = np.asarray(im) #image 转矩阵
        #添加到X数组里
		X.append(mat)

    #创建一个h5的文件，这个文件的作用是成为后续训练或者测试的数据集
	file = h5py.File("dataset//test.h5","w")
    #分别将特征数据X与该数据代表的含义即标签 转为numpy数组保存到h5文件里
	file.create_dataset('X', data=np.array(X))
	file.create_dataset('Y', data=np.array(Y))
    #关闭流文件
	file.close()



if __name__ == "__main__":
    win32api.MessageBox(0, "即将进入数据数据采集阶段", "提醒", win32con.MB_OK)
    takePhoto()
    win32api.MessageBox(0, "即将进行手势切割阶段", "提醒", win32con.MB_OK)
    getHand()
    win32api.MessageBox(0, "即将进行手势图片压缩处理阶段", "提醒", win32con.MB_OK)
    resize_img()
    win32api.MessageBox(0, "即将进行图片扩充（各种翻转以获取更多特征数据）阶段", "提醒", win32con.MB_OK)
    rotate_main()
    win32api.MessageBox(0, "即将进行数据转存为h5文件阶段", "提醒", win32con.MB_OK)
    image_to_h5()



