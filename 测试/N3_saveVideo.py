# 请使用python编程实现从笔记本电脑摄像头捕获视频程序并保存在本地
import cv2
import win32api, win32con


def saveVideo():
    win32api.MessageBox(0, "请按键盘q键结束录制！", "提醒", win32con.MB_OK)
    #0是拍 1是读
    cap = cv2.VideoCapture(0)
    #用来设置需要保存视频的格式 指定编码器
    forucc = cv2.VideoWriter_fourcc(*'XVID')
    # 文件路径 编码器 fps帧率 要保存的画面尺寸
    out = cv2.VideoWriter('./dataset/new_pic/out.avi',forucc,20,(640,480))
    while cap.isOpened():   #摄像头打开的情况下。
        ret,frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 2)
        cv2.rectangle(frame, (150, 150), (450, 450), (0, 255, 0))
        if ret == True:
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    win32api.MessageBox(0, "录制结束，请点击'识别动态手势'按钮进行识别！", "提醒", win32con.MB_OK)


if __name__ == "__main__":
    saveVideo()