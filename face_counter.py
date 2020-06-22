#coding=utf-8
import cv2
import numpy as np
from PIL import Image

# 設置參數
overlay_enable = False
blur_pixel = (4, 4)
move_detection_thresh = 64
move_min_size = 2500
face_detection_times = 3
face_min_pixel = (120, 120)
face_detection_interval = 500
avg_adjustment = 0.02

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

# 載入面部檢測
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

# 面部檢測暫存
face_detected = []
face_detected_backup = []
face_count = 0
is_counted = False
move_detected = []

# 載入覆蓋圖片
image_path = 'overlay.png'
overlay_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
image_width, image_height = Image.open(image_path).size

def overlay(background, image, alpha=1.0, x=0, y=0, scale=1.0, width=0, height=0):
    (h, w) = background.shape[:2]
    background = np.dstack([background, np.ones((h, w), dtype="uint8") * 255])
    overlay = cv2.resize(image, None,fx=width/image_width,fy=height/image_height)
    (wH, wW) = overlay.shape[:2]
    output = background.copy()
    try:
        if x<0 : x = w+x
        if y<0 : y = h+y
        if x+wW > w: wW = w-x  
        if y+wH > h: wH = h-y
        overlay=cv2.addWeighted(output[y:y+wH, x:x+wW],alpha,overlay[:wH,:wW],1.0,0)
        output[y:y+wH, x:x+wW ] = overlay
    except Exception as e:
        print(e)
    output= output[:,:,:3]
    return output

# 面部檢測區域重複率
def is_coincide(box1, box2):
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False

def solve_coincide(box1, box2):
    if is_coincide(box1,box2)==True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col=min(x02,x12)-max(x01,x11)
        row=min(y02,y12)-max(y01,y11)
        intersection=col*row
        area1=(x02-x01)*(y02-y01)
        area2=(x12-x11)*(y12-y11)
        coincide=intersection/(area1+area2-intersection)
        return coincide
    else:
        return False

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
  move_detected = []
  face_detected = []

  for c in cnts:
    # 忽略太小的區域
    if cv2.contourArea(c) < move_min_size :
        continue

    # 計算等高線的外框範圍
    (x, y, w, h) = cv2.boundingRect(c)
    move_detected = move_detected + [(x, y, w, h)]
    
    # 畫出外框
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  for (old_x, old_y, old_w, old_h) in face_detected_backup:
    for (move_x, move_y, move_w, move_h) in move_detected:
        rate = solve_coincide((old_x, old_y, old_w, old_h), (move_x, move_y, move_w, move_h))
        if rate == False:
            continue
        if old_x > move_x : old_x = move_x
        if old_y > move_y : old_y = move_y
        if old_h < move_h : old_x = move_h
        if old_w < move_w : old_x = move_w
        face_detected = face_detected + [(old_x, old_y, old_w, old_h)]
        break

  # 畫出等高線
  cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2) 

  # 面部檢測
  faces = face_cascade.detectMultiScale(gray, 1.1, face_detection_times, cv2.CASCADE_SCALE_IMAGE, face_min_pixel)

  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 顯示面部覆蓋圖片
    if overlay_enable:
        frame = overlay(frame, overlay_image, x=x, y=y, width=w, height=h)

    is_counted = False
    for (old_x, old_y, old_w, old_h) in face_detected_backup:
        rate = solve_coincide((x, y, w, h), (old_x, old_y, old_w, old_h))
        if rate == False:
            continue
        elif rate > 0:
            is_counted = True
            break
    if is_counted == False:
        face_count = face_count + 1
    face_detected = face_detected + [(x, y, w, h)]

  face_detected_backup = face_detected

  # 顯示面部檢測數
  cv2.putText(frame, "Total count: " + str(face_count), (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)

  # 顯示偵測結果影像
  cv2.imshow('frame', frame)

  if cv2.waitKey(face_detection_interval) & 0xFF == ord('q'):
    break

  # 更新平均影像
  cv2.accumulateWeighted(blur, avg_float, avg_adjustment)
  avg = cv2.convertScaleAbs(avg_float)

webcam.release()
cv2.destroyAllWindows()
