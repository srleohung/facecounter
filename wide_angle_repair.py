import cv2
import numpy as np

# 廣角有效區域截取
def cut(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x,y,w,h = cv2.boundingRect(cnts)
    r = max(w/ 2, h/ 2)
    # 提取有效區域
    img_valid = img[y:y+h, x:x+w]
    return img_valid, int(r)
 
# 廣角矯正
def undistort(src,r):
    # r： 半徑， R: 直徑
    R = 2 * r

    # Pi: 圓周率
    Pi = np.pi

    # 存儲映射結果
    dst = np.zeros((R, R, 3))
    src_h, src_w, _ = src.shape
 
    # 圓心
    x0, y0 = src_w//2, src_h//2
 
    # 數組， 循環每個點
    range_arr = np.array([range(R)])
 
    theta = Pi - (Pi/R)*(range_arr.T)
    temp_theta = np.tan(theta)**2
 
    phi = Pi - (Pi/R)*range_arr
    temp_phi = np.tan(phi)**2
 
    tempu = r/(temp_phi + 1 + temp_phi/temp_theta)**0.5
    tempv = r/(temp_theta + 1 + temp_theta/temp_phi)**0.5
 
    # 用於修正正負號
    flag = np.array([-1] * r + [1] * r)
 
    # 加0.5是為了四捨五入求最近點
    u = x0 + tempu * flag + 0.5
    v = y0 + tempv * np.array([flag]).T + 0.5
 
    # 防止數組溢出
    u[u<0]=0
    u[u>(src_w-1)] = src_w-1
    v[v<0]=0
    v[v>(src_h-1)] = src_h-1
 
    # 插值
    dst[:, :, :] = src[v.astype(int),u.astype(int)]
    return dst
 
if __name__ == "__main__":
    # 讀取圖片
    frame = cv2.imread('frame.jpeg')

    # 截取有效區域
    cut_img,R = cut(frame)
    cv2.imwrite('frame_cuted.jpeg',cut_img)

    # 修正廣角區域
    result_img = undistort(cut_img,R)
    cv2.imwrite('frame_repaired.jpeg',result_img)