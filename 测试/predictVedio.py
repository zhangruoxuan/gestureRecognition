# 导入所需要的库
import os
from PIL import Image
import numpy as np
import h5py
import tensorflow.compat.v1 as tf

import imageio
from skimage import img_as_ubyte
import cv2
import predictPicture as pre
import xlrd#操控Excel表
# import getHand


# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)

#处理视频 得到相应的帧数据集
# 处理视频 得到相应的帧数据集
# timeF设置每多少帧取一次图片，默认为25
# 一般1秒钟的视频大概在二十多到上百帧，画质越好一秒的帧数越多，电视机一般都是25到30帧一秒
def dealVedio(timeF = 25):
    # 读取视频文件
    videoCapture = cv2.VideoCapture("dataset/new_pic/out.avi")
    # 通过摄像头的方式
    # videoCapture=cv2.VideoCapture(1)
    # 读帧
    success, frame = videoCapture.read()
    #i代表当前帧数
    i = 0
    #timeF设置每多少帧取一次图片
    timeF = 5
    #取得图片数
    j = 0
    while success:
        i = i + 1
        if (i % timeF == 0):
            j = j + 1
            #第一个参数是保存的当前帧  第二个参数是目录 第三个参数是图片张数：会命名为0_1，0_2这种
            #如果设置常数 比如1 则指挥生成最后一张图片 代表的是截取的最后一张
            #咱们是按照截取最后一张图来识别，方便磁盘管理，只保留一张图（该方法运行速度慢，改为批处理）
            save_image(frame, './img/vedioout/0_', j)
            #cv2.imshow('1',frame)
            #print('save image:', j)

        success, frame = videoCapture.read()



#该方法是用来删除r'./img/vedioout'文件下的jpg文件
def deleteDir():
    # 指明被遍历的文件夹
    rootdir = r'./img/vedioout'
    for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
        for filename in filenames:
            if os.path.splitext(filename)[1]==".jpg":
                currentPath = os.path.join(parent, filename)
                os.remove(currentPath)
                #print('已删除：',currentPath)
    print('已经清除缓存...')

#以下方法和ProducePicture里的一样，可以选择调用（前提是得改变图片路径，或者改写代码用传参的方式调用）
# def getHand():
#     # 指明被遍历的文件夹
#     rootdir = r'./img/vedioout'
#     for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
#         for filename in filenames:
#             if os.path.splitext(filename)[1]==".jpg":
#                 currentPath = os.path.join(parent, filename)
#
#                 img = Image.open(currentPath)
#
#                 box1 = (150,150,450, 450)  # 设置左、上、右、下的像素
#                 image1 = img.crop(box1)  # 图像裁剪
#                 image1.save(r"./img/vedioout/" + filename)  # 存储裁剪得到的图像
#                 image1.save(r"./img/vedioout2/" + filename)  # 存储裁剪得到的图像

#压缩图片,把图片压缩成64*64的
def resize_img():
	dirs = os.listdir("./img/gethand")
	for filename in dirs:
		if not filename.endswith(".jpg"):
			continue
        #获取文本操作句柄，类似于python提供的文本操作open()函数，filename是要打开的文件名，mode是以何种方式去读写，将会返回一个文本操作句柄。
		im = tf.gfile.GFile("./img/gethand//{}".format(filename), 'rb').read()
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

			imageio.imwrite("./img/resize//{}".format(filename),img_as_ubyte(resized_im))


#图片转h5文件
def image_to_h5():
    #遍历文件夹下所有图片
	dirs = os.listdir("./img/resize")
	Y = [] #label标签
	X = [] #data数据
	# print(len(dirs))
	for filename in dirs:
        # print(filename);
		if not filename.endswith(".jpg"):
			continue
        # X.append(cv2.imread("./img/resize//{}".format(filename)))
    # filenamereturn X
    #     #将1-1,2-1这种切割成1,2这种，代表的是手势编码，也就是标签
		label = int(filename.split('_')[0])
		# print(filename)
        #添加到Y数组里
		Y.append(label)
        #打开图片
		im = Image.open("./img/resize//{}".format(filename)).convert('RGB')
        #将图片转换为矩阵数组，即训练或者测试用的特征数据

        # print(filename)
        #
		# mat = np.asarray(im) #image 转矩阵
        #添加到X数组里
		X.append( np.asarray(im))

    #创建一个h5的文件，这个文件的作用是成为后续训练或者测试的数据集
	file = h5py.File("./img/vedioout//vedioout.h5","w")
    #分别将特征数据X与该数据代表的含义即标签 转为numpy数组保存到h5文件里
	file.create_dataset('X', data=np.array(X))
	file.create_dataset('Y', data=np.array(Y))
    #关闭流文件
	file.close()


def getHand():
    e = []  # 测试
    imagenum = 0  # 存的时候名字
    for n in range(1,2):
        disad = './img/vedioout'
        vlist = os.listdir(disad)
        number = len(vlist)
        disad+='/0_'
        ad1 = disad + '1.jpg'  # 第一张是基准
        images = []
        images.append(ad1)
        for num in range(2, number ):  # 把每个文件夹剩下的都存images
            imagead = disad + str(num) + '.jpg'  # 地址
            # print(disad+str(num)+'.jpg')
            images.append(imagead)

        raw = []  # 皮肤检测后的图片
        p = 0
        t=0
        for i in images:  # 对每一个图片皮肤检测

            t+=1
            img = Image.open(i)
            box1 = (150, 150, 450, 450)  # 设置左、上、右、下的像素
            image111 = img.crop(box1)  # 图像裁剪
            image111.save(r"./img/cut/0_" + str(t)+'.jpg')  # 存储裁剪得到的图像

        disad = './img/cut'
        vlist = os.listdir(disad)
        number = len(vlist)
        disad += '/0_'
        ad1 = disad + '1.jpg'  # 第一张是基准
        images = []
        images.append(ad1)
        for num in range(2, number ):  # 把每个文件夹剩下的都存images
            imagead = disad + str(num) + '.jpg'  # 地址
            # print(disad+str(num)+'.jpg')
            images.append(imagead)

        raw = []  # 皮肤检测后的图片
        i=0
        for i in images:  # 对每一个图片皮肤检测

            if type(i) != str or i == '':
                p = 1
                break
            img = cv2.imread(i, cv2.IMREAD_COLOR)  # 读图片
            # 皮肤检测
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)  #
            (y, cr, cb) = cv2.split(ycrcb)
            cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
            _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            res = cv2.bitwise_and(img, img, mask=skin)
            kernel = np.ones((5, 5), np.uint8)
            erode = cv2.erode(res, kernel)  # 腐蚀
            res = cv2.dilate(erode, kernel)
            # res = cv2.cvtColor(res, cv2.COLOR_BGR2YCR_CBGRAY)
            raw.append(res)  # 存储皮肤检测后的图片
            imagenum += 1
            if imagenum<10:
                imgroad = './img/gethand/0_0' + str(imagenum) + '.jpg'
            else:
                imgroad = './img/gethand/0_' + str(imagenum) + '.jpg'
            cv2.imwrite(imgroad, res)
        if p == 1:
            continue  # 防止报错
        # for n in range(1, len(raw)):
        #     gray_diff = cv2.absdiff(raw[0], raw[n])  # 2-图片减去第一张
        #     _, thresh1 = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)  # 二值，大于40的为255，小于0
        #     _, thresh2 = cv2.threshold(raw[n], 40, 255, cv2.THRESH_BINARY)  # 吧关键帧转成二值
        #     binary = cv2.bitwise_and(thresh1, thresh2)  # 与运算
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #     erode = cv2.erode(binary, kernel)  # 腐蚀
        #     erode = cv2.erode(erode, kernel)  # 腐蚀
        #     erode = cv2.erode(erode, kernel)  # 腐蚀
        #     imagen = cv2.GaussianBlur(erode, (5, 5), 0)  # 高斯滤波
        #     e.append(imagen)
        #     x, y, w, h = cv2.boundingRect(imagen)  # 找到坐标
        #     image = cv2.imread(images[n])
        #     image = image[y - 10:y + h, x:x + w]  # 截取
        #     imagenum += 1
        #     imgroad = './img/gethand/0_' + str(imagenum) + '.jpg'
        #     if image.size == 0:
        #         continue
        #     if w * h > 65000:
        #         continue
        #     if w * h < 2000:
        #         continue
        #     if w < 50:
        #         continue
        #     cv2.imwrite(imgroad, image)
        #     print(w * h)
        #     print(imgroad)




#识别视频中的帧  得到两个数据finalResult为字符串数组 addResult为整句话
def recognitionVedio():
    #先对视频进行处理
    dealVedio()
    # 以下方法是将得到的该图片进行预处理 处理成64*64那种格式
    getHand()  # 取手
    # getHand()
    resize_img()  # 变成64*64格式
    image_to_h5()  # 转为h5数据格式

    # 接下来就是对该图片进行识别
    url = "./img/vedioout//vedioout.h5"
    # 加载一个模型 即 继承我们之前训练的那个模型结构，然后用来预测新的数据
    parameters = pre.load_parameters()
    # 通过读入h5文件 获得图片的特征数据X以及标签数据y，此时用来预测没有用到y
    X_train, y_train = pre.loadData(url)
    # 预测新的数据 新数据为temp
    result = pre.predict(parameters, X_train)
    #print(result)
    parameters.clear()
    #以下是得到的序列转为所对应的文字信息
    # print('result的类型',type(result))
    # 将numpy.ndarray转为list 上面通过识别以后为numpy.ndarray格式的数据，要想去邻重 需要转为list格式
    result = result.tolist()
    # print('result的类型',type(result))
    a = [0, 0, 0, 0]
    res = []
    for i in range(len(result) - 1):
        a[result[i]] += 1
        if result[i + 1] != result[i]:
            if a[result[i]] > 1:
                if len(res) > 0 and res[-1] != result[i] or len(res) == 0:
                    res.append(result[i])
            a[result[i]] = 0
    if result[-1] == result[-2]:
        res.append(result[-1])
    # print("res:",res)
    result=res
    print('未去重复：',result)
    # 去邻重复
    result = del_adjacent(result)
    print('未转化为文字的标签序列为：',result)
    #以下方法是将序列转为文字描述
    #先读取excel表中的数据
    cols = read_excel()
    #定义一个 文字描述的数组
    # print("==",cols)
    finalResult = []
    #index是下标索引 item是值
    for index, item in enumerate(result):
        #将序列转为文字，并存入数组中
        # print("index",index)
        # print(int(result[index]))
        msg = cols[int(result[index])-1]#或者这样写 msg = cols[int(item)]
        # print("msg", msg)
        finalResult.append(msg)
    #print(finalResult)
    #这个是数组里所以元素合并在一块儿的字符串
    addResult = ''
    for value in finalResult:
        addResult += value
    #print(addResult)

    print('识别的序列为：',finalResult)
    print('识别的结果为：',addResult)
    return finalResult,addResult

#该方法用来去重
def del_adjacent(alist):
    #三个参数 第一个是数组长度减一（即最大索引值），第二个是0（即第一个索引值）第三个是步长
    #即从最后一个数据开始往后走，每次走一个 如果当前数据与下一个一样就删除当前数据
    for i in range(len(alist) - 1, 0, -1):
        if alist[i] == alist[i - 1]:
            del alist[i]
    return alist

#该方法用来读取文件中的手势编码所对应的含义
def read_excel():
    # 打开文件 注意：这里可能有些机器放入1.xlsx文件读不出来，可以将1.xlsx在Excel里转为1.xls再读数据就可以了
    workBook = xlrd.open_workbook('./1.xlsx')
    ## 2.1 法1：按索引号获取sheet内容
    sheet1_content1 = workBook.sheet_by_index(0)# sheet索引从0开始
    # 4. 获取整行和整列的值（数组）
    #rows = sheet1_content1.row_values(3) # 获取第四行内容
    # print(rows)
    #获取手势指代的意思，在第二列
    cols = sheet1_content1.col_values(1) # 获取第2列内容
    #掐去表头 剩下的就是手势编码所对应的含义
    #['你好', '好的', '正确', '再见', '谢谢', '走', '我', '爱', '中国']编码是[1,2,3,4,5,6,7,8,9]但是在数组里下标从0开始数不是从1
    cols.remove(cols[0])
    # print(cols)
    return cols

if __name__ == '__main__':
    #del_files('./img/vedioout/')
    finalResult,addResult=recognitionVedio()

    print('识别的序列为：',finalResult)
    print('识别的结果为：',addResult)