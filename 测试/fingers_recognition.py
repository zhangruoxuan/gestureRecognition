from tkinter import *
from PIL import ImageTk
import pandas as pd
from PIL import Image
import easygui
import win32api, win32con  # 终端输入命令：pip install pywin32
from tkinter import ttk
import os
import cv2
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
# import cnn2
# import N1_takePhoto
# import N2_getHand
import N3_saveVideo
# import N4_img2h5
import cnn
# import N6_syRec
# import N6_syRec_2
# import encoding_library
import threading
import imutils
import imageio
import time
import predictVedio
import predictPicture
# import findHandzrx

root = Tk()
root.title('手势识别教学平台')
# 设置窗口大小
root.geometry('900x400+100+100')

pos_x_pre= 180
pos_x_later=510
pos_x_human=180
pos_y=35
path = 'start'
num = 0
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)
# canvas_pre = Canvas(root,width=300,height=300)
# canvas_pre.place(x=pos_x_pre,y=pos_y)
#
# canvas_later = Canvas(root,width=300,height=300,bg = 'white')
# canvas_later.place(x=pos_x_later,y=pos_y)
#
#
# canvas_ren = Canvas(root,width=300,height=300,bg = 'white')
# canvas_ren.place(x=pos_x_human,y=pos_y)


canvas_ren_later = Canvas(root,width=300,height=300,bg = 'white')
canvas_ren_later.place(x=pos_x_pre,y=pos_y)


canvas_saveVideo = Canvas(root,width=300,height=300,bg = 'white')
canvas_saveVideo.place(x=pos_x_later,y=pos_y)

# Label(root,text='识别前手势演示'.encode('utf-8').decode('utf-8'), font=('楷体', 18)).place(x=260,y=5)


Label(root,text='识别后手势演示'.encode('utf-8').decode('utf-8'), font=('楷体', 18)).place(x=260,y=337)

Label(root,text='录制的视频'.encode('utf-8').decode('utf-8'), font=('楷体', 18)).place(x=560,y=337)


# 初始化是识别结果
var = StringVar()
label = Label( root, textvariable=var,  font=('楷体', 20),relief=RAISED )
var.set(0)
label.place(x=30,y=280)

recognition_result=[]
class videoPlayer_later():
    def __init__(self):
        self.frame = None
        self.thread = None
        self.stopEvent = None

        # initialize the root window and image panel

        self.panel = None

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

    def videoLoop(self):
        # keep looping over frames until we are instructed to stop

        for i in recognition_result:
            if i != '':
                var.set(i)
                video_path = './virtual_human/'+str(i)+'.avi'
                video = imageio.get_reader(video_path, 'ffmpeg')
                for frame in video:
                    self.frame = imutils.resize(frame, width=300)
                    image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    image = image[0:300, 0:430]
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)

                    # if the panel is not None, we need to initialize it
                    if self.panel is None:
                        self.panel = Label(image=image)
                        self.panel.image = image
                        self.panel.place(x=180,y=35)
                        # otherwise, simply update the panel
                    else:
                        self.panel.configure(image=image)
                        self.panel.image = image
                    time.sleep(0.02)



# def show_human_later(result):
#     path = './virtual_human/' + str(result) + '.avi'
#     print(path)
#     cap2 = cv2.VideoCapture(path)
#     while (cap2.isOpened()):
#         _, frame1 = cap2.read()
#         frame1 = imutils.resize(frame1, width=300)
#
#         roi = frame1[0:300, 0:300]
#         img_pre = Image.fromarray(roi)
#         img_pre = ImageTk.PhotoImage(img_pre)
#         canvas_ren_later.create_image(0, 0, anchor=NW, image=img_pre)
#
#         root.update_idletasks()
#         root.update()
#         cv2.waitKey(30)
#     # cap2.release()



class show_saveVideo():
    def __init__(self):
        self.frame = None
        self.thread = None
        self.stopEvent = None

        # initialize the root window and image panel

        self.panel = None

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

    def videoLoop(self):
        # keep looping over frames until we are instructed to stop
        video_path = './dataset/new_pic/out.avi'
        video = imageio.get_reader(video_path, 'ffmpeg')
        for frame in video:
            self.frame = imutils.resize(frame, width=640)
            # image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = self.frame[150:450, 150:450]
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

                    # if the panel is not None, we need to initialize it
            if self.panel is None:
                self.panel = Label(image=image)
                self.panel.image = image
                self.panel.place(x=510,y=35)
                        # otherwise, simply update the panel
            else:
                self.panel.configure(image=image)
                self.panel.image = image
            time.sleep(0.02)




frame = Frame(root)

frame.place(x=880, y=50, width=800, height=300)




df = pd.read_excel('1.xlsx')
# print(df)
for i in df.index.values:  #获取行号的索引，并对其进行遍历
    # 根据i来获取每一行指定的数据，并利用to_dict方法转成字典
    row_data = df.iloc[i].to_dict()
    # print(row_data['手势编码'])
    # print(row_data)

code_num = []
code_name = []
code_1 = []
code_2 = []
code_3 = []

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
    rootdir = r'./img/cut'
    for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
        for filename in filenames:
            if os.path.splitext(filename)[1]==".jpg":
                currentPath = os.path.join(parent, filename)
                os.remove(currentPath)

    rootdir = r'./img/gethand'
    for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
        for filename in filenames:
            if os.path.splitext(filename)[1] == ".jpg":
                currentPath = os.path.join(parent, filename)
                os.remove(currentPath)

    rootdir = r'./img/resize'
    for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
        for filename in filenames:
            if os.path.splitext(filename)[1] == ".jpg":
                currentPath = os.path.join(parent, filename)
                os.remove(currentPath)

    print('已经清除缓存...')



def emmmm():
    finalResult, addResult= predictVedio.recognitionVedio()
    # print(addResult)
    # recognition_result=finalResult
    # for i in finalResult:
        # show_human_later(t)
    global recognition_result
    recognition_result=finalResult
    videoPlayer_later()
    print(recognition_result)
    return finalResult



Button(root, text="录制识别手势",width = 12,command=N3_saveVideo.saveVideo).place(x=30,y=80)

Button(root, text="播放录制手势",width = 12,command=show_saveVideo).place(x=30,y=130)

Button(root, text="识别动态手势",width = 12,command=emmmm).place(x=30,y=180)
Button(root, text="清除缓存",width = 12,command=deleteDir).place(x=30,y=230)


menuBar = Menu(root)
root.configure(menu=menuBar)

fileMenu = Menu(menuBar)
jihebianhuanMenu = Menu(menuBar)

menuBar.add_cascade(label="文件", menu=fileMenu)
# fileMenu.add_command(label="打开摄像头", command=open_camera)


menuBar.add_cascade(label="第三章 图像几何变换", menu=jihebianhuanMenu)
root.mainloop()
