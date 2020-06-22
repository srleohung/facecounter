#coding=utf-8
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# 設置參數
interval = 200
specific_object_name = "person"
specific_object_minimum_confidence = 0.75

# 構造參數解析並解析參數
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 初始化MobileNet SSD類標籤列表
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
# 生成標籤邊界框顏色
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 加載序列化模型
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# 初始化視頻
vs = VideoStream(src=0).start()
# 攝像頭傳感器預熱
time.sleep(2.0)
# 初始化FPS計數器
fps = FPS().start()

# 循環播放視頻流中的幀
while True:
	# 從視頻流中抓取幀並調整其大小
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# 抓取框架尺寸並將其轉換為Blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# 使Blob通過網絡並獲得檢測結果
	net.setInput(blob)
	detections = net.forward()

	# 循環檢測
	for i in np.arange(0, detections.shape[2]):
		# 提取與以下內容相關的準確值
		confidence = detections[0, 0, i, 2]

		# 忽略太小準確值
		if confidence > args["confidence"]:
			# 計算（x，y）坐標對象的邊界框
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# 選擇特定對象
			label_name = CLASSES[idx]
			label_confidence = confidence
			if label_name == specific_object_name and label_confidence > specific_object_minimum_confidence:
				# 在框架上繪製
				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# 顯示偵測結果影像
	cv2.imshow("Frame", frame)

	if cv2.waitKey(interval) & 0xFF == ord('q'):
		break

	# 更新FPS計數器
	fps.update()

# 停止計時器
fps.stop()

cv2.destroyAllWindows()
vs.stop()