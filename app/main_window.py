import cv2
import os
import warnings
import numpy as np
import torch
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import QEvent, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from app.dialogs import show_message
from app.generated_ui import Ui_mainWindow
from app.theme import apply_theme, set_model_status, set_pause_button
from config import YOLO_FACE_WEIGHTS
from inference import CONFIG
from inference.predictor import build_transform, detect_faces_and_emotions
from models.networks import coatnet_1
from models.yolov5_face.detect_face import detect, load_model

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MainWindow(QMainWindow, Ui_mainWindow):
    EMOTION_LABELS = {
        "angry": "愤怒",
        "disgust": "厌恶",
        "fear": "恐惧",
        "happy": "高兴",
        "neutral": "平静",
        "sad": "悲伤",
        "surprise": "惊讶",
    }

    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 初始化UI
        apply_theme(self)

        self.image_files = []
        self.current_image_id = -1
        self.cap = None
        self.timer = None
        self.is_paused = False
        self.models_ready = False
        self._preview_pixmap = None

        self.current_comboBox_type = self.comboBox.currentText()
        self.comboBox.currentIndexChanged.connect(self.update_combo_type)  # 连接信号：当选项变化时自动更新

        self.lineEdit.mousePressEvent = self.open_file_dialog  # 设置lineEdit点击事件
        self.lineEdit.textChanged.connect(self.update_start_button_state)
        self.pushButton_start.clicked.connect(self.on_start_clicked)  # 设置pushButton_start点击事件
        self.pushButton_pause.clicked.connect(self.toggle_pause)
        self.pushButton_right.clicked.connect(self.right_clicked)  # 设置pushButton_right点击事件
        self.pushButton_left.clicked.connect(self.left_clicked)  # 设置pushButton_left点击事件
        self.pushButton_left.setEnabled(False)
        self.pushButton_right.setEnabled(False)
        self.label_picture.installEventFilter(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_model = None
        self.emotion_model = None
        self.pushButton_start.setEnabled(False)
        self.reset_preview()
        QTimer.singleShot(0, self.load_models)

    def load_models(self):
        """加载人脸检测和表情分类模型"""
        set_model_status(self, "正在加载模型", "loading")
        try:
            face_weights_path = str(YOLO_FACE_WEIGHTS)
            self.face_model = load_model(face_weights_path, device=self.device)
            print("人脸检测模型：", face_weights_path)

            self.emotion_model = coatnet_1(num_classes=len(CONFIG["classes"]))
            self.emotion_model = self.emotion_model.to(self.device)
            self.emotion_model.eval()
            ckpt_path = str(CONFIG["checkpoint_path"])
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.emotion_model.load_state_dict(checkpoint["model_state_dict"])
            print("表情分类模型：", ckpt_path)
        except Exception as exc:
            self.models_ready = False
            set_model_status(self, "模型加载失败", "error")
            self.update_start_button_state()
            show_message(
                self,
                "模型加载失败",
                f"无法加载识别模型，请检查权重文件。\n\n{exc}",
                "error"
            )
            return

        self.models_ready = True
        set_model_status(self, f"模型就绪 · {str(self.device).upper()}", "ready")
        self.update_start_button_state()

    def update_combo_type(self):
        """选项变化时更新当前类型"""
        self.current_comboBox_type = self.comboBox.currentText()
        self.lineEdit.clear()
        self.stop_capture()
        self.image_files = []
        self.current_image_id = -1
        self.pushButton_left.setEnabled(False)
        self.pushButton_right.setEnabled(False)
        show_navigation = self.current_comboBox_type == "图片"
        self.pushButton_left.setVisible(show_navigation)
        self.pushButton_right.setVisible(show_navigation)
        self.pushButton_pause.setVisible(not show_navigation)
        self.pushButton_pause.setEnabled(False)
        set_pause_button(self, False)
        self.reset_preview()
        if self.current_comboBox_type == "图片":
            self.lineEdit.setEnabled(True)
            self.lineEdit.setPlaceholderText("点击选择图片文件夹")
        elif self.current_comboBox_type == "视频":
            self.lineEdit.setEnabled(True)
            self.lineEdit.setPlaceholderText("点击选择视频文件")
        else:
            self.lineEdit.setEnabled(False)
            self.lineEdit.setPlaceholderText("无需选择路径，点击开始识别")
        self.update_start_button_state()

    def update_start_button_state(self):
        """根据模型状态和输入路径更新开始按钮。"""
        if not self.models_ready:
            self.pushButton_start.setEnabled(False)
            return
        if self.current_comboBox_type == "摄像头":
            self.pushButton_start.setEnabled(True)
            return
        self.pushButton_start.setEnabled(bool(self.lineEdit.text().strip()))

    def open_file_dialog(self, event):  # line edit
        event.accept()
        if self.current_comboBox_type == "图片":
            dir_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
            if dir_path:
                self.lineEdit.setText(dir_path)
        elif self.current_comboBox_type == "视频":
            video_filter = "视频文件 (*.mp4 *.avi *.mov *.mkv)"  # 设置视频文件类型过滤器
            path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", '', filter=video_filter)
            if path:
                self.lineEdit.setText(path)
        else:
            pass

    def on_start_clicked(self):
        """启动按钮的槽函数"""
        if not self.models_ready:
            show_message(self, "模型未就绪", "识别模型尚未加载完成", "warning")
            return

        input_type = self.current_comboBox_type
        input_path = self.lineEdit.text().strip()

        if input_type == "图片":
            if not os.path.isdir(input_path):
                show_message(self, "路径错误", "请选择有效的图片文件夹", "error")
                return
            self.image_files = sorted([
                f for f in os.listdir(input_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            if not self.image_files:
                show_message(self, "没有图片", "所选文件夹中没有 JPG 或 PNG 图片", "warning")
                return
            self.stop_capture()
            self.current_image_id = 0
            self.pushButton_left.setEnabled(True)
            self.pushButton_right.setEnabled(True)
            self.update_label_pictures(os.path.join(input_path, self.image_files[0]))
            return

        if input_type == "视频":
            if not os.path.isfile(input_path) or not self.is_video_file(input_path):
                show_message(self, "路径错误", "请选择有效的视频文件", "error")
                return
            self.update_label_video(input_path)
            return

        self.update_label_camera()

    def toggle_pause(self):
        """暂停或继续视频与摄像头识别。"""
        if self.timer is None:
            return

        self.is_paused = not self.is_paused
        if self.is_paused:
            self.timer.stop()
            self.previewMeta.setText("识别已暂停")
        else:
            self.timer.start(30)
            mode = "摄像头" if self.current_comboBox_type == "摄像头" else "视频"
            self.previewMeta.setText(f"{mode}识别中")
        set_pause_button(self, self.is_paused)

    def right_clicked(self):
        if self.current_comboBox_type == '图片' and self.image_files:
            self.current_image_id = self.current_image_id + 1
            if self.current_image_id < len(self.image_files):
                image_path = os.path.join(self.lineEdit.text().strip(), self.image_files[self.current_image_id])
                self.update_label_pictures(image_path)
            else:
                self.current_image_id = self.current_image_id - 1
                show_message(self, "提示", "已是最后一张图片", "info")
        else:
            pass

    def left_clicked(self):
        if self.current_comboBox_type == '图片' and self.image_files:
            self.current_image_id = self.current_image_id - 1
            if self.current_image_id > -1:
                image_path = os.path.join(self.lineEdit.text().strip(), self.image_files[self.current_image_id])
                self.update_label_pictures(image_path)
            else:
                self.current_image_id = self.current_image_id + 1
                show_message(self, "提示", "已是第一张图片", "info")
        else:
            pass

    def is_video_file(self, path):
        """验证是否为视频文件（扩展名检查）"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        return os.path.splitext(path)[1].lower() in video_extensions

    def update_label_pictures(self, image_path):
        file_name = os.path.basename(image_path)
        self.label_file.setText(file_name)
        self.previewMeta.setText("正在识别")
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = detect_faces_and_emotions(
                self.face_model,
                self.emotion_model,
                self.device,
                image_path
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.set_preview_pixmap(self.cv2_to_qpixmap_picture(result['annotated_image']))
        self.update_result_display(result["faces"])
        face_count = len(result["faces"])
        face_text = f"{face_count} 张人脸" if face_count else "未检测到人脸"
        total = len(self.image_files)
        self.previewMeta.setText(f"{self.current_image_id + 1}/{total} · {face_text}")

    def update_result_display(self, faces):
        """把识别结果以更易读的中文和百分比显示。"""
        if not faces:
            self.label_outcome.setText("未检测到人脸")
            self.label_confidence.setText("--")
            self.confidenceBar.setValue(0)
            return

        labels = [
            self.EMOTION_LABELS.get(face["emotion"], face["emotion"])
            for face in faces
        ]
        confidences = [self.normalize_confidence(face["confidence"]) for face in faces]
        max_confidence = max(confidences)

        self.label_outcome.setText(" · ".join(labels))
        if len(faces) == 1:
            self.label_confidence.setText(f"{max_confidence * 100:.1f}%")
        else:
            self.label_confidence.setText(f"最高 {max_confidence * 100:.1f}%")
        self.confidenceBar.setValue(round(max_confidence * 100))

    @staticmethod
    def normalize_confidence(confidence):
        value = float(confidence)
        return value / 100 if value > 1 else value

    def reset_preview(self):
        self._preview_pixmap = None
        self.label_picture.clear()
        self.label_picture.setText("选择输入并点击“开始识别”")
        self.previewMeta.setText("尚未载入")
        self.label_file.setText("未选择文件")
        self.reset_result()

    def reset_result(self):
        self.label_outcome.setText("等待识别")
        self.label_confidence.setText("--")
        self.confidenceBar.setValue(0)

    def set_preview_pixmap(self, pixmap):
        self._preview_pixmap = QPixmap(pixmap) if pixmap is not None else None
        if self._preview_pixmap is None or self._preview_pixmap.isNull():
            self.label_picture.clear()
            return
        self.label_picture.setText("")
        self.resize_preview()

    def resize_preview(self):
        if self._preview_pixmap is None or self._preview_pixmap.isNull():
            return
        target_size = self.label_picture.size()
        target_size.setWidth(max(1, target_size.width() - 20))
        target_size.setHeight(max(1, target_size.height() - 20))
        scaled = self._preview_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label_picture.setPixmap(scaled)

    def eventFilter(self, source, event):
        if source is self.label_picture and event.type() == QEvent.Resize:
            self.resize_preview()
        return super().eventFilter(source, event)

    def stop_capture(self):
        self.is_paused = False
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        set_pause_button(self, False)

    def update_label_video(self, video_path):
        """初始化视频处理并启动定时器"""
        self.stop_capture()

        # 初始化视频捕获
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            show_message(self, "无法打开视频", f"无法打开视频文件：{video_path}", "error")
            return

        self.label_file.setText(os.path.basename(video_path))
        self.previewMeta.setText("视频处理中")
        self.reset_result()
        self.is_paused = False
        set_pause_button(self, False)
        self.pushButton_pause.setEnabled(True)

        # 创建并启动定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_label_camera(self):
        """初始化摄像头处理并启动定时器"""
        self.stop_capture()

        # 初始化摄像头（默认使用设备索引 0）
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            show_message(self, "无法打开摄像头", "请检查摄像头是否已连接或被其他程序占用。", "error")
            return

        self.label_file.setText("摄像头 0")
        self.previewMeta.setText("摄像头实时识别")
        self.reset_result()
        self.is_paused = False
        set_pause_button(self, False)
        self.pushButton_pause.setEnabled(True)

        # 创建并启动定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        """更新视频帧"""
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            # 处理当前帧
            result = self.process_frame(frame)
            if result:
                self.set_preview_pixmap(self.cv2_to_qpixmap_video(result['annotated_image']))
                self.update_result_display(result["faces"])
                face_count = len(result["faces"])
                prefix = "摄像头" if self.current_comboBox_type == "摄像头" else "视频"
                self.previewMeta.setText(f"{prefix} · {face_count} 张人脸" if face_count else f"{prefix} · 未检测到人脸")
        else:
            # 视频播放结束
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.previewMeta.setText("视频播放结束")
            self.pushButton_pause.setEnabled(False)
            self.pushButton_pause.setText("播放结束")
            show_message(self, "播放完成", "视频已播放完毕。", "success")

    def process_frame(self, frame):
        """处理单帧"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            detection_results = detect(
                model=self.face_model,
                source=frame_rgb,
                device=self.device
            )
        except Exception as e:
            print(f"Error in detect: {e}")
            return None

        if not detection_results:
            print("No detection results returned")
            return {"annotated_image": frame.copy(), "faces": []}

        final_faces = []
        for frame_data in detection_results:
            for face in frame_data["faces"]:
                x1, y1, x2, y2 = [int(c) for c in face["bbox"]]
                # 确保裁剪区域有效
                y1, y2 = max(0, y1), min(frame_rgb.shape[0], y2)
                x1, x2 = max(0, x1), min(frame_rgb.shape[1], x2)
                if y2 <= y1 or x2 <= x1:
                    continue  # 跳过无效区域
                face_region = frame_rgb[y1:y2, x1:x2]
                try:
                    emotion, confidence = self.classify_emotion(face_region)
                    final_faces.append({
                        "position": [x1, y1, x2, y2],
                        "emotion": emotion,
                        "confidence": confidence
                    })
                except Exception as e:
                    print(f"Error in classify_emotion: {e}")
                    continue

        annotated = self.draw_annotations(frame.copy(), final_faces)
        return {
            "annotated_image": annotated,
            "faces": final_faces
        }

    def classify_emotion(self, face_region):
        """表情分类"""
        transform = build_transform(CONFIG["img_size"])
        img_pil = Image.fromarray(face_region)
        tensor = transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.emotion_model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        pred_idx = np.argmax(probs)
        return CONFIG["classes"][pred_idx], float(probs[pred_idx])

    def draw_annotations(self, frame, faces):
        """绘制标注"""
        accent = (143, 182, 28)
        scale = max(0.55, min(1.0, frame.shape[1] / 1400))
        thickness = max(1, round(scale * 2))
        for face in faces:
            x1, y1, x2, y2 = face["position"]
            confidence = self.normalize_confidence(face["confidence"])
            label = f"{face['emotion']} {confidence * 100:.1f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), accent, thickness)
            text_size, baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                thickness
            )
            text_width, text_height = text_size
            text_y = max(y1, text_height + 12)
            cv2.rectangle(
                frame,
                (x1, text_y - text_height - 10),
                (x1 + text_width + 12, text_y + baseline - 6),
                accent,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 6, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                thickness
            )
        return frame

    def cv2_to_qpixmap_video(self, cv_image):
        """将OpenCV图像转换为QPixmap"""
        if not cv_image.flags['C_CONTIGUOUS']:
            cv_image = np.ascontiguousarray(cv_image)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)

    def cv2_to_qpixmap_picture(self, cv_img: np.ndarray) -> QPixmap:
        """将 OpenCV 的 numpy.ndarray 转换为 PyQt 的 QPixmap"""
        # 确保图像是连续内存（避免转换时出现对齐问题）
        if not cv_img.flags['C_CONTIGUOUS']:
            cv_img = np.ascontiguousarray(cv_img)

        # 转换颜色通道：BGR → RGB（OpenCV默认是BGR，PyQt需要RGB）
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # 获取图像尺寸和通道信息
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w  # 每行的字节数

        # 创建 QImage 对象（直接引用数据内存，避免复制）
        qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 转换为 QPixmap
        return QPixmap.fromImage(qimage)

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        self.stop_capture()
        event.accept()
