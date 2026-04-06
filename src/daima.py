import cv2
import os
import numpy as np

# 检查 OpenCV 版本
print(f"OpenCV 版本: {cv2.__version__}")

# 使用 OpenCV 内置的 LBPH 人脸识别器
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("LBPH 识别器创建成功")
except AttributeError:
    print("错误：OpenCV 未安装 contrib 模块，请执行: pip install opencv-contrib-python")
    exit()

# 加载人脸检测器
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(face_cascade_path)

if detector.empty():
    print(f"错误：无法加载级联分类器 {face_cascade_path}")
    exit()

# 训练数据路径
path = 'training_data'

if not os.path.exists(path):
    os.makedirs(path)
    print(f"\n请把训练照片放入 {path} 文件夹")
    print("照片命名格式: 人名.jpg (例如: 张三.jpg)")
    exit()


def get_images_and_labels(path):
    """加载图片并提取人脸"""
    image_paths = [os.path.join(path, f) for f in os.listdir(path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if not image_paths:
        print(f"错误：{path} 文件夹中没有图片")
        return None, None, None

    face_samples = []
    ids = []
    id_map = {}
    current_id = 0

    print(f"\n找到 {len(image_paths)} 张图片，正在处理...")

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        name = os.path.splitext(filename)[0]

        if name not in id_map:
            id_map[name] = current_id
            current_id += 1

        person_id = id_map[name]

        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"  跳过无法读取的文件: {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 0:
            print(f"  警告：{filename} 中未检测到人脸")
            continue

        # 取最大的人脸（假设每张图主要人物）
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y + h, x:x + w]

        face_samples.append(face_roi)
        ids.append(person_id)
        print(f"  ✓ {filename} -> {name} (ID: {person_id})")

    # 创建反向映射 ID->Name
    id_to_name = {v: k for k, v in id_map.items()}
    return face_samples, ids, id_to_name


# 训练模型
print("=" * 50)
faces, ids, id_to_name = get_images_and_labels(path)

if not faces:
    print("没有可用的人脸数据，程序退出")
    exit()

print(f"\n训练模型中，共 {len(faces)} 个人脸样本...")
recognizer.train(faces, np.array(ids))
print(f"训练完成！可识别人员: {list(id_to_name.values())}")
print("=" * 50)

# 开始实时识别
print("启动摄像头...")
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("错误：无法打开摄像头")
    print("提示：")
    print("  1. 检查摄像头是否被其他程序占用")
    print("  2. 如果是笔记本，确认摄像头未被物理关闭")
    print("  3. 尝试修改代码中的摄像头索引: cv2.VideoCapture(1)")
    exit()

print("摄像头已启动，按 'q' 退出，按 's' 保存当前帧")
print("=" * 50)

frame_count = 0
while True:
    ret, img = cam.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    frame_count += 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 绘制人脸框
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 识别
        face_roi = gray[y:y + h, x:x + w]
        person_id, confidence = recognizer.predict(face_roi)

        # 计算置信度（越小越像）
        if confidence < 70:
            name = id_to_name.get(person_id, "Unknown")
            color = (0, 255, 0)  # 绿色
            confidence_text = f"{round(100 - confidence)}%"
        elif confidence < 100:
            name = id_to_name.get(person_id, "Unknown") + "?"
            color = (0, 165, 255)  # 橙色
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"
            color = (0, 0, 255)  # 红色
            confidence_text = "Low"

        # 显示名字和置信度
        label = f"{name} | {confidence_text}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 显示帧率
    cv2.putText(img, f"Frames: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Face Recognition - Python 3.13', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f'screenshot_{frame_count}.jpg', img)
        print(f"已保存截图: screenshot_{frame_count}.jpg")

cam.release()
cv2.destroyAllWindows()
print("\n程序已安全退出")
