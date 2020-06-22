#coding=utf-8
import cv2
import numpy as np
from PIL import Image

# 設置參數
blur_pixel = (2, 2)
move_detection_thresh = 12
face_detection_interval = 30
avg_adjustment = 0

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

# 載入背景圖片
background = Image.open("background.jpg")

# 調整平均值
for x in range(100):
  ret, frame = webcam.read()
  if ret == False:
    break
  blur = cv2.blur(frame, blur_pixel)
  cv2.accumulateWeighted(blur, avg_float, 0.05)
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

  # 清除和原始值相同的位置
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
  frame[:, :, 3] = thresh
  cv2.imwrite('frame.png', frame)

  # 生成背景及合成影格至背景
  frame = Image.open("frame.png")
  new_frame = background.copy()
  new_frame.paste(frame, (0, 0), frame)
  new_frame.save("frame.jpg")
  reload_frame = cv2.imread('frame.jpg', cv2.IMREAD_UNCHANGED)

  # 顯示偵測結果影像
  cv2.imshow('frame', reload_frame)

  if cv2.waitKey(face_detection_interval) & 0xFF == ord('q'):
    break

  # 更新平均影像
  cv2.accumulateWeighted(blur, avg_float, avg_adjustment)
  avg = cv2.convertScaleAbs(avg_float)

webcam.release()
cv2.destroyAllWindows()
