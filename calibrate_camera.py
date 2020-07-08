import cv2
import numpy as np

def calibrate_single(imgNums, CheckerboardSize, Nx_cor, Ny_cor):

    # 單目(普通+廣角/魚眼)攝像頭標定
    # :param imgNums: 標定所需樣本數,一般在20~40之間.按鍵盤空格鍵實時拍攝
    # :param CheckerboardSize: 標定的棋盤格尺寸,必須為整數.(單位:mm或0.1mm)
    # :param Nx_cor: 棋盤格橫向內角數
    # :param Ny_cor: 棋盤格縱向內角數
    # :return mtx: 內參數矩陣.{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}
    # :return dist: 畸變係數.(k_1,k_2,p_1,p_2,k_3)

    # 找棋盤格角點(角點精準化迭代過程的終止條件)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, CheckerboardSize, 1e-6)  # (3,27,1e-6)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE  # 11
    flags_fisheye = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW  # 14
 
    # 世界坐標系中的棋盤格點,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((1, Nx_cor * Ny_cor, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:Nx_cor, 0:Ny_cor].T.reshape(-1, 2)
 
    # 儲存棋盤格角點的世界坐標和圖像坐標對
    # 在世界坐標系中的三維點
    objpoints = []
    # 在圖像平面的二維點
    imgpoints = []
 
    # 用來標誌成功檢測到的棋盤格畫面數量
    count = 0
 
    while (True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 尋找棋盤格模板的角點
        ok, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), flags)
        if count >= imgNums:
            break
        # 如果找到，添加目標點，圖像點
        if ok:
            objpoints.append(objp)
            # 獲取更精確的角點位置
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners)

            # 將角點在圖像上顯示
            cv2.drawChessboardCorners(frame, (Nx_cor, Ny_cor), corners, ok)
            count += 1
            print('Find the total number of board corners: ' + str(count))
        
        cv2.waitKey(1)

    global mtx, dist
 
    # 標定。 rvec和tvec是在獲取了相機內參mtx，dist之後通過內部調用solvePnPRansac（）函數獲得的
    # ret為標定結果，mtx為內參數矩陣，dist為畸變係數，rvecs為旋轉矩陣，tvecs為平移向量
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[:2][::-1], None, criteria
    )

    # 攝像頭內參mtx = [[f_x，0，c_x] [0，f_y，c_y] [0,0,1]]
    print('mtx=np.array( ' + str(mtx.tolist()) + " )")
    # 畸變係數dist =（k1，k2，p1，p2，k3）
    print('dist=np.array( ' + str(dist.tolist()) + " )")
 
    # 魚眼/大廣角鏡頭的單目標定
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    RR = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
    TT = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, gray.shape[:2][::-1], K, D, RR, TT, flags_fisheye, criteria
    )
    # 攝像頭內參，此結果與mtx分類更為穩定和精確
    print("K=np.array( " + str(K.tolist()) + " )")
    # 畸變係數D =（k1，k2，k3，k4）
    print("D=np.array( " + str(D.tolist()) + " )")
    # 計算反投影誤差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Calculate back projection total error: ", mean_error / len(objpoints))
 
    cv2.destroyAllWindows()
    return mtx, dist, K, D

if __name__ == "__main__":
    # 開啟網路攝影機
    cap = cv2.VideoCapture(0)

    mtx, dist, K, D = calibrate_single(30, 27, 9, 6)
    # mtx=np.array( [[525.1974085051108, 0.0, 322.46321668550206], [0.0, 470.6897728780676, 207.1415778240149], [0.0, 0.0, 1.0]] )
    # dist=np.array( [[-0.5440259736780028], [0.4582542025510915], [-0.004460196250793969], [-0.010744165783903798], [-0.31459559977372276]] )
    # K=np.array( [[508.94954778109036, 0.0, 308.80041433072194], [0.0, 453.8659706150624, 201.00963020768984], [0.0, 0.0, 1.0]] )
    # D=np.array( [[-0.1710816455825023], [-0.046660635179406704], [0.3972574493629046], [-0.3102470529709773]] )

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    p = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), None)
    mapx2, mapy2 = cv2.fisheye.initUndistortRectifyMap(K, D, None, p, (width, height), cv2.CV_32F)
    while (True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        frame_rectified = cv2.remap(frame, mapx2, mapy2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        cv2.imshow('frame_rectified', frame_rectified)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()