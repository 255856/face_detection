import cv2
import numpy as np
import os
from datetime import datetime

def enhance_image(image):#图像增强处理
    # 自适应直方图均衡化
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # 锐化处理
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

def detect_faces1(image):# Haar级联（调整参数）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_haar = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # 减小缩放步长
        minNeighbors=7,  # 增加邻居数
        minSize=(50, 50),  # 增大最小尺寸
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces_haar
def  detect_faces2(image):# ：DNN检测
    dnn_faces = []
    try:
        net = cv2.dnn.readNetFromCaffe(
            "models/deploy.prototxt",
            "models/res10_300x300_ssd_iter_140000.caffemodel")
        if not os.path.exists("deploy.prototxt") or not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
            print("警告: 未找到DNN模型文件，将仅使用Haar级联检测器")
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # 使用CUDA加速
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # 使用GPU
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # 置信度阈值
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                dnn_faces.append(box.astype("int"))
    except:
        pass
    # 合并结果（优先使用DNN结果）
    return dnn_faces

def mark_features(image, faces):#精确标记面部特征
    marked = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in faces:
        # 绘制人脸矩形
        cv2.rectangle(marked, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 在人脸ROI内检测特征
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = marked[y:y + h, x:x + w]
        # 眼睛检测（使用更精确的左右眼分类器）
        eye_cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml'),
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml'),
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
        ]
        for cascade in eye_cascades:
            eyes = cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
        # 鼻子检测（调整参数）
        nose_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
        noses = nose_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(30, 30))
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 1)
    return marked

def save_results(image, enhanced, marked1,marked2,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # 原始图（带直方图）
    hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    hist_img = np.zeros((150,image.shape[1],3,),dtype=np.uint8)
    cv2.normalize(hist, hist, 0, hist_img.shape[0], cv2.NORM_MINMAX)
    for i in range(1, 256):
        cv2.line(hist_img, (i - 1, 150 - int(hist[i - 1][0])),
                 (i, 150 - int(hist[i][0])), (255, 255, 255), 1)
    combined = np.vstack([image, hist_img])
    cv2.imwrite(os.path.join(output_dir, "01_original_with_hist.jpg"), combined)
    cv2.imwrite(os.path.join(output_dir, "02_enhanced.jpg"), enhanced)# 增强后的图像
    cv2.imwrite(os.path.join(output_dir, "03_marked1.jpg"), marked1)# 带标记的图像
    cv2.imwrite(os.path.join(output_dir, "04_marked2.jpg"), marked2)  # 带标记的图像
    print(f"结果已保存到：{os.path.abspath(output_dir)}")
    print("成功保存3张图片！")

def main(input_path):
    # 读取图像
    image = cv2.imread(input_path)
    if image is None:
        print(f"错误：无法读取图像 {input_path}")
        return
    #  图像增强
    enhanced = enhance_image(image)
    #  人脸检测
    faces1 = detect_faces1(enhanced)
    if len(faces1) == 0:
        print("警告：未检测到人脸，尝试原始图像检测...")
        faces1 = detect_faces1(image)  # 回退到原始图像检测
    faces2 = detect_faces2(enhanced)
    if len(faces2) == 0:
        print("警告：未检测到人脸，尝试原始图像检测...")
        faces2 = detect_faces2(image)  # 回退到原始图像检测
    # 特征标记
    marked1 = mark_features(enhanced, faces1)
    marked2 = mark_features(enhanced, faces2)
    # 保存结果
    output_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_results(image, enhanced, marked1,marked2,output_dir)
    # 显示原始图像和处理后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Equalized Histogram', enhanced)
    cv2.imshow('Enhanced Image1', marked1)
    cv2.imshow('Enhanced Image2', marked2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True

def video():
    # 初始化检测器Haar
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # 打开摄像头/视频文件
    print("1.摄像头")
    print("2.视频（路径）")
    x=int(input("请输入(esc退出视频)："))
    while x!=1 or x!=2:
        if x == 1:
            cap = cv2.VideoCapture(0)  # 0为默认摄像头
            break
        elif x == 2:
            path = input("请输入视频路径：").strip('"')  # 去除路径双引号
            path = path.replace("\\ ", "/").strip()  # 处理反斜杠
            if not os.path.exists(path):
                print(f"错误：输入视频路径 {path} 不存在")
            else:
                cap = cv2.VideoCapture(path)
                break
        else:
            print("输入错误！")
            x=int(input("请重新输入："))
    # 设置可调整大小的窗口
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Detection', 800, 600)  # 初始尺寸
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 人脸检测，使用Haar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# 转换为灰度图
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)# 人脸检测
        # faces=detect_faces2(frame)# 使用DNN模型：
        # mark=mark_features(frame, faces)
        for (x, y, w, h) in faces:# 绘制检测框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('Face Detection', frame)# 显示结果
        if cv2.waitKey(1) == 27:# 按ESC退出
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("*" * 60)
    print("                        人脸检测系统")
    print("*" * 60)
    print("1.图片人脸检测")
    print("2.视频人脸检测")
    print("3.退出程序")
    num=int(input("请输入："))
    while num!=1 or num!=2 or num!=3:
        if num == 1:
            path = input("请输入图片路径：").strip('"')  # 去除路径双引号
            path = path.replace("\\ ", "/").strip()  # 处理反斜杠
            if not os.path.exists(path):
                print(f"错误：输入图片 {path} 不存在")
            else:
                main(path)
                print("*" * 60)
                print("1.图片人脸检测")
                print("2.视频人脸检测")
                print("3.退出程序")
                num = int(input("请输入："))
        elif num == 2:
            video()
            print("*" * 60)
            print("1.图片人脸检测")
            print("2.视频人脸检测")
            print("3.退出程序")
            num = int(input("请输入："))
        elif num == 3:
            print("程序已退出！")
            print("*" * 60)
            break
        else:
            print("输入错误,请重新输入！")
            num = int(input("请输入："))
