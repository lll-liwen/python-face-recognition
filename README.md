# python-face-recognition
基于人脸识别的项目

项目介绍：基于 OpenCV LBPH 算法的轻量级人脸识别项目，原文件因要下dlib，但我们由于 Python 3.13 暂不支持 dlib，我们使用 OpenCV 替代方案，如果之前装过 opencv-python，先卸载pip uninstall opencv-python -y。
安装包含人脸识别模块的完整版
pip install opencv-contrib-python numpy

使用说明
训练阶段
程序启动后会自动：
扫描 training_data/ 目录
检测人脸并提取特征
训练 LBPH 模型
显示可识别的人员列表
实时识别
绿色框+名字：识别成功 (>70% 置信度)
橙色框+名字?：可能匹配 (70-100% 置信度)
红色框+Unknown：未识别 (>100% 置信度)
按键控制：
q：退出程序
s：保存当前截图
