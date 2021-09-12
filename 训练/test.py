import numpy as np
import cv2
cap = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(*'MJPG')
fps = 20.0
frameSize = (640, 480)
out = cv2.VideoWriter('video.avi', codec, fps, frameSize)
print("按键Q-结束视频录制")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()