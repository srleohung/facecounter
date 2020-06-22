# Face Counter
The package attempts to count faces using face detection and motion detection. It will be used on vending machines to calculate product attention and get traffic at that location.

# Usage
Use face_counter.py
```bash
python face_counter.py
```
Use motion_detection.py
```bash
python motion_detection.py
```
Use transparent.py
```bash
python transparent.py
```
Use object_detection.py
```bash
python object_detection.py --prototxt data/MobileNetSSD/MobileNetSSD_deploy.prototxt.txt --model data/MobileNetSSD/MobileNetSSD_deploy.caffemodel
```

## Parameters
Overlay image enable (Boolean)
```
overlay_enable = True
```
Pixel blurring (Tuple)
```
blur_pixel = (4, 4)
```
Move detection thresh (Int) range: 1 - 255
```
move_detection_thresh = 64
```
Minimum size of moving filter (Int) range: 0 - width*height
```
move_min_size = 2500
```
Face detection times (Int) range: > 1
```
face_detection_times = 3 
```
Minimum pixel of face filter (Tuple)
```
face_min_pixel = (120, 120)
```
Face detection interval (Int) range: > 1
```
face_detection_interval = 1
```
Motion detection average adjustment (Float) 
```
avg_adjustment = 0.2
```
Webcam frame width (Int)
```
width = 1280
```
Webcam frame height (Int)
```
height = 960
```