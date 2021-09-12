'''
    例5_1:只需要选择图片即可
    例5_2:只需要选择图片即可
    例5_3:只需要选择图片即可
    例5_4:只需要选择图片即可
    例5_5:请输入1或者2，1代表原始直方图2代表正规化后直方图
    例5_6:请输入1或者2，1代表原始直方图2代表正规化后直方图
    例5_7:请输入1或者2或者3，1代表均衡化后图片2代表原始直方图3代表均衡化后直方图
    例5_8:请输入1或者2或者3，1代表均衡化后图片2代表原始直方图3代表均衡化后直方图
    例5_9:请输入1或者2或者3，1代表均衡化后图片2代表原始直方图3代表均衡化后直方图
'''

from tkinter import *   #导入全部，图形界面
from tkinter import messagebox#弹出框
from PIL import ImageTk, Image, ImageEnhance #图像处理
import cv2 as cv      # pip install opencv-python
import math
import numpy as np #数据处理
import easygui
import matplotlib.pyplot as plt  # 导入绘图模块
import os
root = Tk()
root.title("数字图像处理课程案例演示")
#root.geometry('1500x700')
root.geometry('1300x600+110+50')    #窗口大小和初始位置
#选择的图片地址
sFilePath = 'None'


def buttonClick():
    global img_label_2, photo2
    # 读取图片
    img2 = Image.open('result.png')
    photo2 = ImageTk.PhotoImage(img2)
    img_label_2.configure(image=photo2)

#把代码展示在左边的图框里
def showImage(image):
    global img_label_1,photo
    # 读取图片
    img = Image.open(image)
    print(image)
    (x, y) = img.size  # read image size
    # 固定图片大小 设置长与高
    x_s = 500  # 设置长度
    y_s = 300  # 设置高度
    out = img.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
    # 保存原图片，避免因为地址而加载不出来   并设置原图的当前根目录地址
    #out.save('original.png')
    photo = ImageTk.PhotoImage(out)
    img_label_1.configure(image=photo)

def open_camera():
    print('进入摄像头')
    global img_label_1, photo_3
    cap = cv.VideoCapture(0)  # 获取摄像头设备或打开摄像头
    if cap.isOpened():  # 判断摄像头是否已经打开,若打开则进入循环
        # cap = cv.VideoCapture(0)
        codec = cv.VideoWriter_fourcc(*'MJPG')
        fps = 20.0
        frameSize = (640, 480)
        out = cv.VideoWriter('video.avi', codec, fps, frameSize)
        print("按键Q-结束视频录制")
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                out.write(frame)
                cv.imshow('Q=quit', frame)
                if cv.waitKey(1) == ord('q'):
                    break
            else:
                break
    cap.release()
    out.release()
    cv.destroyAllWindows()



    #     while True:  # 无无限循环
    #         ret, frame = cap.read()  # cap返回两个值,所以要用两个变量接收,即使ret获取视频播放状态,
    #         # frame获取视频的一帧,即使ret不需要用也要接收,不然cap返回两
    #         # 个值却用一个变量接收程序就会报错,根据python的语法,ret接收
    #         # 第一个返回的值,frame接收第二个返回的值
    #         cv.imshow('camera', frame)  # 显示图片,显示视频是通过连续显示一张张图片来实现的
    #         if cv.waitKey(1) & 0xff == ord('q'):  # 如果在循环中按下键盘的q键
    #             cv.imwrite('original.png', frame) # 将最后一帧写入当前工程文件的目录下,名
    #             im = Image.open('original.png')
    #             (x, y) = im.size  # read image size
    #             x_s = 300  # define standard width
    #             y_s = 300  # calc height based on standard width
    #             out = im.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
    #             out.save('original.png')
    #             img_3 = Image.open('original.png')
    #             photo_3 = ImageTk.PhotoImage(img_3)
    #             img_label_1.configure(image=photo_3)
    #             break
    # cap.release()  # 释放资源,即销毁进程
    # cv.destroyAllWindows()  # 销毁所有窗口

# def open_image():
#     global img_label_1, photo_3,sFilePath
#     try:
#         #读取图片
#         sFilePath = easygui.fileopenbox()
#         img_3 = Image.open(sFilePath)
#         print(sFilePath)
#         (x, y) = img_3.size  # read image size
#         #固定图片大小 设置长与高
#         x_s = 400  # 设置长度
#         y_s = 400  # 设置高度
#         out = img_3.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
#         #保存原图片，避免因为地址而加载不出来   并设置原图的当前根目录地址
#         out.save('original.png')
#         sFilePath = 'original.png'
#         # 设置图片流模式 并将图片加载在img_label_1中
#         photo_3 = ImageTk.PhotoImage(out)
#         img_label_1.configure(image=photo_3)
#     except:
#         messagebox.showinfo("提示", "您未选择图片，请打开例题并选择图片！")



def way5_1():
    #open_image()
    open_camera()
    #处理图片
    image = cv.imread("original.png")  # 导入一幅图像
    hist = cv.calcHist([image], [0], None, [256], [0, 255])  # 计算其统计直方图信息
    print(hist[-15:])  # 输出统计直方图信息，为一维数组最后15个
    strs = "image=" + str(hist[-15:])
    board = np.asarray([[255 for i in range(500)] for j in range(400)], dtype='uint8')
    for i, txt in enumerate(strs.split('\n')):
        y = 30 + i * 30
        cv.putText(board, txt, (120, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, 0)
    cv.imwrite("result.png", board)



''' 以下是界面布局以及组件设置  # fg前景色 bg背景色'''
# Label(root,fg='red',text='提示：请直接选择章节例题并选择相应图片，点击运行得到结果', font=('楷体', 15),
#               width=80, height=2).pack()
#左边视图
img_1 = Image.open('6_123.png')
photo_1 = ImageTk.PhotoImage(img_1)  # 转化成可以在tkinter用的图片。在root实例化创建，否则会报错
img_label_1 = Label(root, image=photo_1)
img_label_1.pack(side='left')
#右边视图
img_2 = Image.open('6_123.png')
photo_2 = ImageTk.PhotoImage(img_2)
img_label_2 = Label(root, image=photo_2)
img_label_2.pack(side='right')

Button(root, text="打开摄像头", command=way5_1).pack(side='top')
Button(root, text="建立训练集", command=way5_1).pack(side='top')
Button(root, text="打开摄像头", command=way5_1).pack(side='top')


e = StringVar()
xz__text = Entry(root, bd=5, textvariable=e)
xz__text.pack(side=BOTTOM)
e.set('1')
xz__textBiaoZhi = 1

# xz_label = Label(root, fg='red',text='例1到5:只需要选择图片即可',font=('楷体', 12))
# xz2_label = Label(root, fg='red',text='例5_6:输入1代表原始直方图\n2代表正规化后直方图',font=('楷体', 12))
# xz3_label = Label(root, fg='red',text='例7到9:输入1代表均衡化后图片2\n代表原始直方图3代表均衡化后直方图',font=('楷体', 12))

# xz3_label.pack(side=BOTTOM)
# xz2_label.pack(side=BOTTOM)
# xz_label.pack(side=BOTTOM)



mainmenu = Menu(root)
# 确定主菜单
root.config(menu=mainmenu)
# 在主菜单栏上创建一个不显示分窗口的子菜单栏submain1
submenu1 = Menu(mainmenu, tearoff=0)
submenu1.add_separator() #添加一个分割线
submenu1.add_command(label='退出', command=root.destroy)

# 主菜单栏上添加依次添加三个有多层级的子菜单栏
# mainmenu.add_cascade(label='第五章案例演示', menu=submenu1)

root.mainloop()