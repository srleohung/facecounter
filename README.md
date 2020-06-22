# Face Counter
The package attempts to count faces using face detection and motion detection. It will be used on vending machines to calculate product attention and get traffic at that location.

# Install
```
pip install -r requirements.txt
```

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

# Parameters
Overlay image enable (Boolean)
```python
overlay_enable = True
```
Pixel blurring (Tuple)
```python
blur_pixel = (4, 4)
```
Move detection thresh (Int) range: 1 - 255
```python
move_detection_thresh = 64
```
Minimum size of moving filter (Int) range: 0 - width*height
```python
move_min_size = 2500
```
Face detection times (Int) range: > 1
```python
face_detection_times = 3 
```
Minimum pixel of face filter (Tuple)
```python
face_min_pixel = (120, 120)
```
Face detection interval (Int) range: > 1
```python
face_detection_interval = 1
```
Motion detection average adjustment (Float) 
```python
avg_adjustment = 0.2
```
Webcam frame width (Int)
```python
width = 1280
```
Webcam frame height (Int)
```python
height = 960
```