from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from app.theme import APP_STYLE


DIALOG_STYLE = """
QDialog {
    background: transparent;
}

QFrame#dialogShell {
    background: #FFFFFF;
    border: 1px solid #DDE5EF;
    border-radius: 10px;
}

QLabel#dialogBadge {
    border-radius: 18px;
    font-size: 15px;
    font-weight: 700;
}

QLabel#dialogBadge[kind="info"] {
    background: #EAF1FF;
    color: #1D4ED8;
}

QLabel#dialogBadge[kind="success"] {
    background: #E5F7F2;
    color: #087A63;
}

QLabel#dialogBadge[kind="warning"] {
    background: #FFF4D6;
    color: #9A6700;
}

QLabel#dialogBadge[kind="error"] {
    background: #FDECEC;
    color: #B42318;
}

QLabel#dialogTitle {
    color: #111827;
    font-size: 16px;
    font-weight: 700;
}

QLabel#dialogMessage {
    color: #475569;
    font-size: 13px;
}

QPushButton#dialogClose {
    background: transparent;
    border: none;
    color: #64748B;
    font-size: 15px;
    font-weight: 700;
    padding: 0;
}

QPushButton#dialogClose:hover {
    background: #F1F5F9;
    color: #0F172A;
}

QPushButton#dialogPrimary {
    background: #2563EB;
    border: none;
    color: #FFFFFF;
    font-weight: 700;
    min-width: 92px;
    padding: 0 18px;
}

QPushButton#dialogPrimary:hover {
    background: #1D4ED8;
}

QPushButton#dialogPrimary:pressed {
    background: #1E40AF;
}
"""


class MessageDialog(QDialog):
    BADGE_TEXT = {
        "info": "i",
        "success": "OK",
        "warning": "!",
        "error": "x",
    }

    def __init__(self, parent, title, message, kind="info"):
        super().__init__(parent)
        self._drag_position = None
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        self.setMinimumWidth(450)
        self.setStyleSheet(APP_STYLE + DIALOG_STYLE)

        kind = kind if kind in self.BADGE_TEXT else "info"
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(18, 18, 18, 18)

        shell = QFrame(self)
        shell.setObjectName("dialogShell")
        outer_layout.addWidget(shell)

        shadow = QGraphicsDropShadowEffect(shell)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 7)
        shadow.setColor(QColor(15, 23, 42, 55))
        shell.setGraphicsEffect(shadow)

        content = QVBoxLayout(shell)
        content.setContentsMargins(22, 20, 22, 20)
        content.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(11)

        badge = QLabel(self.BADGE_TEXT[kind], shell)
        badge.setObjectName("dialogBadge")
        badge.setProperty("kind", kind)
        badge.setFixedSize(36, 36)
        badge.setAlignment(Qt.AlignCenter)
        header.addWidget(badge)

        title_label = QLabel(title, shell)
        title_label.setObjectName("dialogTitle")
        header.addWidget(title_label)
        header.addStretch(1)

        close_button = QPushButton("X", shell)
        close_button.setObjectName("dialogClose")
        close_button.setFixedSize(30, 30)
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.clicked.connect(self.reject)
        header.addWidget(close_button)
        content.addLayout(header)

        message_label = QLabel(message, shell)
        message_label.setObjectName("dialogMessage")
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content.addWidget(message_label)

        footer = QHBoxLayout()
        footer.addStretch(1)
        confirm_button = QPushButton("确定", shell)
        confirm_button.setObjectName("dialogPrimary")
        confirm_button.setMinimumHeight(38)
        confirm_button.setCursor(Qt.PointingHandCursor)
        confirm_button.setDefault(True)
        confirm_button.clicked.connect(self.accept)
        footer.addWidget(confirm_button)
        content.addLayout(footer)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_position is not None:
            self.move(event.globalPos() - self._drag_position)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_position = None
        super().mouseReleaseEvent(event)


def show_message(parent, title, message, kind="info"):
    dialog = MessageDialog(parent, title, message, kind)
    dialog.exec_()
