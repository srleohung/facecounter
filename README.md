# Face Counter
The package attempts to count faces using face detection and motion detection. It will be used on vending machines to calculate product attention and get traffic at that location.

# Install
```bash
pip install -r requirements.txt
# Need to download models when using real-time pose estimation.
bash model_download.sh
```

# Usage
## Real-time Face Counter
Use face_counter.py - 
```bash
python face_counter.py
```
## Real-time Motion Detection
Use motion_detection.py
```bash
python motion_detection.py
```
## Real-time Transparent Background
Use transparent.py
```bash
python transparent.py
```
## Real-time Object Detection
Use object_detection.py
```bash
python object_detection.py --prototxt data/MobileNetSSD/MobileNetSSD_deploy.prototxt.txt --model data/MobileNetSSD/MobileNetSSD_deploy.caffemodel
```
## Calibrate Wide-angle Camera
Use calibrate_camera.py
```bash
python calibrate_camera.py
```
## Wide-angle Repair
Use wide_angle_repair.py
```bash
python wide_angle_repair.py
```
## Real-time Pose Estimation
Use pose_estimation.py
- MPI pretrained model
```bash
python pose_estimation.py --proto data/mpi/pose_deploy_linevec_faster_4_stages.prototxt  --model data/mpi/pose_iter_160000.caffemodel --dataset MPI
```
- Body_25 pretrained model
```bash
python pose_estimation.py --proto data/body_25/body_25_deploy.prototxt  --model data/body_25/pose_iter_584000.caffemodel
```
- COCO pretrained model
```bash
python pose_estimation.py --proto data/coco/deploy_coco.prototxt  --model data/coco/pose_iter_440000.caffemodel --dataset COCO
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

# Reference
OpenCV data, from https://github.com/opencv/opencv

MobileNetSSD caffemodel, from https://github.com/PINTO0309/MobileNet-SSD-RealSense

Wide Angle Repair, from https://blog.csdn.net/donkey_1993/article/details/103909811

Calibrate Camera, from https://blog.csdn.net/Thomson617/article/details/103506391

Pose Estimation, from https://github.com/legolas123/cv-tricks.com