from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QStyle

from config import ASSETS_DIR as ASSETS_PATH


ASSETS_DIR = ASSETS_PATH


APP_STYLE = """
QWidget {
    color: #172033;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}

QMainWindow,
QWidget#centralwidget {
    background: #F3F6FA;
}

QLabel#appMark {
    background: transparent;
}

QLabel#headerTitle {
    color: #111827;
    font-size: 22px;
    font-weight: 700;
}

QLabel#headerSubtitle {
    color: #64748B;
    font-size: 12px;
}

QLabel#modelStatus {
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    padding: 0 12px;
}

QLabel#modelStatus[state="loading"] {
    background: #FFF4D6;
    color: #9A6700;
}

QLabel#modelStatus[state="ready"] {
    background: #E5F7F2;
    color: #087A63;
}

QLabel#modelStatus[state="error"] {
    background: #FDECEC;
    color: #B42318;
}

QFrame#controlPanel,
QFrame#previewPanel,
QFrame#resultPanel {
    background: #FFFFFF;
    border: 1px solid #DFE6F0;
    border-radius: 8px;
}

QLabel#label_inputType,
QLabel#previewTitle,
QLabel#resultTitle {
    color: #1E293B;
    font-size: 14px;
    font-weight: 700;
}

QLabel#previewMeta {
    background: #EEF4FF;
    border-radius: 7px;
    color: #2563EB;
    font-size: 12px;
    font-weight: 600;
    padding: 5px 10px;
}

QComboBox,
QLineEdit {
    background: #F8FAFC;
    border: 1px solid #D8E0EA;
    border-radius: 7px;
    color: #1E293B;
    font-size: 13px;
    padding: 0 12px;
    selection-background-color: #BFDBFE;
}

QComboBox:hover,
QLineEdit:hover {
    border-color: #B9C7D8;
}

QComboBox:focus,
QLineEdit:focus {
    background: #FFFFFF;
    border: 1px solid #2563EB;
}

QComboBox {
    padding-right: 38px;
}

QComboBox::drop-down {
    background: #EEF4FF;
    border: none;
    border-left: 1px solid #D8E0EA;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 34px;
}

QComboBox::drop-down:hover {
    background: #E3EDFF;
}

QComboBox QAbstractItemView {
    background: #FFFFFF;
    border: 1px solid #D8E0EA;
    outline: none;
    padding: 5px;
    selection-background-color: #E8F0FF;
    selection-color: #1D4ED8;
}

QPushButton {
    background: #FFFFFF;
    border: 1px solid #D8E0EA;
    border-radius: 7px;
    color: #25324A;
    font-size: 13px;
    font-weight: 600;
    padding: 0 14px;
}

QPushButton:hover {
    background: #F7FAFF;
    border-color: #AFC3DD;
}

QPushButton:pressed {
    background: #EDF3FB;
}

QPushButton:disabled {
    background: #F1F5F9;
    border-color: #E2E8F0;
    color: #94A3B8;
}

QPushButton#pushButton_start {
    background: #2563EB;
    border: none;
    color: #FFFFFF;
    font-weight: 700;
}

QPushButton#pushButton_start:hover {
    background: #1D4ED8;
}

QPushButton#pushButton_start:pressed {
    background: #1E40AF;
}

QPushButton#pushButton_start:disabled {
    background: #AFC5E8;
    color: #EFF6FF;
}

QPushButton#pushButton_pause {
    background: #FFF8E7;
    border: 1px solid #F1D18A;
    color: #8A5A00;
}

QPushButton#pushButton_pause:hover {
    background: #FFF2D2;
    border-color: #E7BE62;
}

QPushButton#pushButton_pause[state="paused"] {
    background: #EAF1FF;
    border-color: #BFD2F4;
    color: #1D4ED8;
}

QPushButton#pushButton_pause[state="paused"]:hover {
    background: #DFE9FF;
    border-color: #9FB8E9;
}

QLabel#label_picture {
    background: #F8FAFC;
    border: 1px dashed #C8D4E3;
    border-radius: 8px;
    color: #7B8798;
    font-size: 15px;
    padding: 10px;
}

QFrame#resultDivider {
    background: #E7ECF3;
    border: none;
}

QLabel#emotionCaption,
QLabel#confidenceCaption,
QLabel#fileCaption {
    color: #64748B;
    font-size: 12px;
    font-weight: 600;
}

QLabel#label_outcome {
    color: #087A63;
    font-size: 29px;
    font-weight: 700;
}

QLabel#label_confidence {
    color: #1D4ED8;
    font-size: 28px;
    font-weight: 700;
}

QLabel#label_file {
    background: #F8FAFC;
    border: 1px solid #E5EAF2;
    border-radius: 7px;
    color: #334155;
    font-size: 12px;
    padding: 9px 10px;
}

QProgressBar#confidenceBar {
    background: #EAF0F7;
    border: none;
    border-radius: 4px;
}

QProgressBar#confidenceBar::chunk {
    background: #14B8A6;
    border-radius: 4px;
}
"""


def apply_theme(window):
    chevron_path = (ASSETS_DIR / "chevron-down.svg").as_posix()
    logo_path = ASSETS_DIR / "logo.svg"
    window.setStyleSheet(
        APP_STYLE
        + "\nQComboBox::down-arrow {"
        + f' image: url("{chevron_path}"); width: 12px; height: 12px; '
        + "}\n"
    )

    style = window.style()
    app_icon = QIcon(str(logo_path))
    window.setWindowIcon(app_icon)
    app = QApplication.instance()
    if app is not None:
        app.setWindowIcon(app_icon)

    logo = QPixmap(str(logo_path))
    window.appMark.setPixmap(
        logo.scaled(
            window.appMark.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    )
    window.pushButton_start.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
    window.pushButton_pause.setIcon(style.standardIcon(QStyle.SP_MediaPause))
    window.pushButton_left.setIcon(style.standardIcon(QStyle.SP_ArrowLeft))
    window.pushButton_right.setIcon(style.standardIcon(QStyle.SP_ArrowRight))
    window.pushButton_start.setIconSize(QSize(15, 15))
    window.pushButton_pause.setIconSize(QSize(14, 14))
    window.pushButton_left.setIconSize(QSize(14, 14))
    window.pushButton_right.setIconSize(QSize(14, 14))

    window.lineEdit.setCursor(Qt.PointingHandCursor)
    window.pushButton_start.setCursor(Qt.PointingHandCursor)
    window.pushButton_pause.setCursor(Qt.PointingHandCursor)
    window.pushButton_left.setCursor(Qt.PointingHandCursor)
    window.pushButton_right.setCursor(Qt.PointingHandCursor)

    set_pause_button(window, False)
    set_model_status(window, "正在加载模型", "loading")


def set_model_status(window, text, state):
    window.modelStatus.setText(text)
    window.modelStatus.setProperty("state", state)
    window.modelStatus.style().unpolish(window.modelStatus)
    window.modelStatus.style().polish(window.modelStatus)
    window.modelStatus.update()


def set_pause_button(window, paused):
    button = window.pushButton_pause
    button.setText("继续识别" if paused else "暂停识别")
    button.setProperty("state", "paused" if paused else "running")
    icon = QStyle.SP_MediaPlay if paused else QStyle.SP_MediaPause
    button.setIcon(window.style().standardIcon(icon))
    button.style().unpolish(button)
    button.style().polish(button)
    button.update()
