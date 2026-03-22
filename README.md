# 人脸检测与特征标记系统

基于 OpenCV 实现的人脸检测与面部特征标记系统，支持图片和视频两种模式，融合 Haar 级联与 DNN（SSD）两种检测算法，并提供图像增强预处理功能。

## 📌 功能特点

-  **图片检测模式**：单张图片的人脸检测与眼睛、鼻子特征标记
-  **视频检测模式**：支持摄像头实时检测和本地视频文件检测
-  **双检测器融合**：优先使用 DNN 深度学习模型（SSD），置信度不足时回退使用 Haar 级联分类器
-  **图像增强**：自适应直方图均衡化（CLAHE）+ 锐化处理，有效提升低光照条件下的检测效果
-  **精确特征标记**：使用多个级联分类器检测眼睛（左眼/右眼）和鼻子，并标记在图像上
-  **结果自动保存**：按时间戳创建文件夹，保存原始图、增强图、检测标记图（Haar 和 DNN 两种结果）
-  **直方图可视化**：在保存结果中附带原始图像的灰度直方图，便于分析图像质量

## 🛠️ 技术栈

| 技术 | 说明 |
|------|------|
| Python 3.8+ | 主要开发语言 |
| OpenCV 4.x | 图像处理、人脸检测、特征标记 |
| NumPy | 数组运算、图像矩阵处理 |
| Caffe | DNN 深度学习模型推理 |

## 📁 项目结构
Face-Detection-System/  
│  
├── face_detection.py # 主程序  
├── requirements.txt # Python 依赖包列表  
├── .gitignore # Git 忽略文件配置  
│  
├── models/ # DNN 模型文件目录（需手动下载）  
│ ├── deploy.prototxt # 模型配置文件  
│ └── res10_300x300_ssd_iter_140000.caffemodel # 预训练权重  
│  
├── results/ # 检测结果输出目录（自动生成）  
│ ├── 20241201_143052/ # 按时间戳命名的结果文件夹  
│ ├── 01_original_with_hist.jpg # 原始图像+直方图  
│ ├── 02_enhanced.jpg # 增强后图像  
│ ├── 03_marked1.jpg # Haar 检测标记结果  
│ └── 04_marked2.jpg # DNN 检测标记结果  
│  
└── samples/ # 测试样例目录（可选）  
  ├── test_image.zip 
  └── test_video.mp4  

## 环境安装
1. 克隆仓库到本地：

git clone https://github.com/cxxy31/Face-Detection-System.git
cd Face-Detection-System

2.安装依赖包
pip install -r requirements.txt

3.下载 DNN 模型文件（如果使用 DNN 检测）：
将 deploy.prototxt 和 res10_300x300_ssd_iter_140000.caffemodel 放入 models/ 目录
如果模型文件不存在，程序会自动降级使用 Haar 检测器

## 🚀 快速开始
运行程序
python face_detection.py  
使用说明  
主菜单：  
************************************************************
**人脸检测系统**
1. 图片人脸检测
2. 视频人脸检测
3. 退出程序
图片检测模式：
************************************************************  
**图片检测模式：**
输入图片路径（支持 .jpg, .png, .jpeg 等格式）  
系统自动进行图像增强、人脸检测、特征标记  
结果自动保存到 results/时间戳/ 目录  
弹出窗口显示原始图像、增强图像和检测结果  
视频检测模式：  

**视频检测模式：**  
选择 1：使用摄像头实时检测  
选择 2：输入本地视频文件路径  
实时显示人脸检测框  
按 ESC 键退出检测  

## 📊 检测算法说明
Haar 级联检测器：  
使用 OpenCV 预训练的 haarcascade_frontalface_default.xml  
参数调优：scaleFactor=1.05, minNeighbors=7, minSize=(50,50)  
优点：速度快，无需额外模型文件  
缺点：对角度、光照变化敏感  
DNN 检测器（SSD + ResNet）  
使用 Caffe 框架的 SSD 模型  
输入尺寸：300×300  
置信度阈值：0.7  
支持 CUDA 加速（自动检测并启用）  
优点：精度高，对遮挡和角度变化鲁棒  
缺点：需要下载模型文件，速度较慢  
图像增强模块  
CLAHE（自适应直方图均衡化）：提高局部对比度  
clipLimit=3.0，tileGridSize=(8,8)  
锐化处理：增强图像边缘  
卷积核：[[0,-1,0], [-1,5,-1], [0,-1,0]]  
面部特征标记  
眼睛：使用 haarcascade_eye.xml、haarcascade_lefteye_2splits.xml、haarcascade_righteye_2splits.xml  
鼻子：使用 haarcascade_mcs_nose.xml  
参数调优：scaleFactor=1.05-1.1，minNeighbors=5-7  

## 📈 性能表现
| 检测器 | 平均速度（CPU）|	准确率 | 适用场景 |
|-------|----------------|---------|----------|
|Haar 级联 |	30-40 FPS	| ~85%	| 实时视频、简单场景|
|DNN (SSD) |	15-20 FPS	| ~92%	| 图片检测、复杂场景|  

注：实际性能取决于 CPU/GPU 配置和图像分辨率

## 🔧 代码结构说明
| 函数名 |	功能描述 |
|-----|-----|
| enhance_image()	| 图像增强（CLAHE + 锐化） |
| detect_faces1()	| Haar 级联人脸检测 |
| detect_faces2()	| DNN 深度学习人脸检测 |
| mark_features()	| 面部特征标记（眼睛、鼻子） |
| save_results()	| 保存检测结果和直方图 |
| main()	| 图片检测主流程 |
| video()	| 视频/摄像头检测流程 |
## 📝 待优化方向
增加更多面部特征点（嘴巴、眉毛、轮廓）  
支持批量图片处  
添加人脸识别功能（识别特定人物）  
优化 DNN 推理速度（TensorRT 加速）  
添加 GUI 图形界面  
支持视频输出保存  
