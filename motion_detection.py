#coding=utf-8
import cv2
import numpy as np
from PIL import Image

# 設置參數
blur_pixel = (3, 3)
move_detection_thresh = 32
move_min_size = 2500
face_detection_interval = 200
avg_adjustment = 0.01
x_error_compensation = 30
y_error_compensation = 30

# 開啟網路攝影機
webcam = cv2.VideoCapture(0)

# 設定擷取影像的尺寸大小
width = 1280
height = 960
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 計算畫面面積
area = width * height

# 初始化平均影像
ret, frame = webcam.read()
avg = cv2.blur(frame, blur_pixel)
avg_float = np.float32(avg)

# 調整平均值
for x in range(10):
  ret, frame = webcam.read()
  if ret == False:
    break
  blur = cv2.blur(frame, blur_pixel)
  cv2.accumulateWeighted(blur, avg_float, 0.10)
  avg = cv2.convertScaleAbs(avg_float)

while(webcam.isOpened()):
  # 讀取一幅影格
  ret, frame = webcam.read()

  # 若讀取至影片結尾，則跳出
  if ret == False:
    break

  # 模糊處理
  blur = cv2.blur(frame, blur_pixel)

  # 計算目前影格與平均影像的差異值
  diff = cv2.absdiff(avg, blur)

  # 將圖片轉為灰階
  gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

  # 篩選出變動程度大於門檻值的區域
  ret, thresh = cv2.threshold(gray, move_detection_thresh, 255, cv2.THRESH_BINARY)

  # 使用型態轉換函數去除雜訊
  kernel = np.ones((5, 5), np.uint8)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

  # 產生等高線
  cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # 初始化
  max_x = width
  max_y = height
  max_x2 = 0
  max_y2 = 0
  detected = False

  for c in cnts:
    # 忽略太小的區域
    if cv2.contourArea(c) < move_min_size :
        continue
    detected = True
    # 計算等高線的外框範圍
    (x, y, w, h) = cv2.boundingRect(c)
    if max_x > x : max_x = x
    if max_y > y : max_y = y
    if max_x2 < x+w : max_x2 = x+w
    if max_y2 < y+h : max_y2 = y+h

  # 畫出等高線
  # cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2) 

  # 忽略框外的區域
  if detected: 
    if max_x - x_error_compensation > 0:
        max_x = max_x - x_error_compensation
    if max_x2 + x_error_compensation < width:
        max_x2 = max_x2 + x_error_compensation
    if max_y - y_error_compensation > 0:
        max_y = max_y - y_error_compensation
    if max_y2 + y_error_compensation < height:
        max_y2 = max_y2 + y_error_compensation
    frame = frame[max_y:max_y2, max_x:max_x2]

  # 顯示偵測結果影像
  cv2.imshow('frame', frame)

  if cv2.waitKey(face_detection_interval) & 0xFF == ord('q'):
    break

  # 更新平均影像
  cv2.accumulateWeighted(blur, avg_float, avg_adjustment)
  avg = cv2.convertScaleAbs(avg_float)

webcam.release()
cv2.destroyAllWindows()
