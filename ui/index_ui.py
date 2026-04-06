# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1000, 650)
        Form.setMinimumSize(600, 450)
        Form.setStyleSheet("background-color: #f5f7fa;")

        # 兼容 QMainWindow 和 QWidget
        if hasattr(Form, 'centralWidget') and callable(Form.centralWidget):
            self._central = QtWidgets.QWidget(Form)
            Form.setCentralWidget(self._central)
        else:
            self._central = Form

        # 根布局
        self.rootLayout = QtWidgets.QVBoxLayout(self._central)
        self.rootLayout.setContentsMargins(0, 0, 0, 0)
        self.rootLayout.setSpacing(0)

        # 顶部标题区
        self.headerWidget = QtWidgets.QWidget(self._central)
        self.headerWidget.setStyleSheet("background: transparent;")
        self.headerLayout = QtWidgets.QVBoxLayout(self.headerWidget)
        self.headerLayout.setContentsMargins(0, 20, 0, 10)
        self.headerLayout.setSpacing(8)

        self.titleLabel = QtWidgets.QLabel(self.headerWidget)
        self.titleLabel.setStyleSheet(
            "color: #2c3e50; font-size: 28px; font-weight: bold; background: transparent;")
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.titleLabel.setObjectName("titleLabel")
        self.headerLayout.addWidget(self.titleLabel)

        self.subTitle = QtWidgets.QLabel(self.headerWidget)
        self.subTitle.setStyleSheet(
            "color: #7f8c8d; font-size: 13px; background: transparent;")
        self.subTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.subTitle.setObjectName("subTitle")
        self.headerLayout.addWidget(self.subTitle)

        self.rootLayout.addWidget(self.headerWidget)

        # 中间内容区
        self.contentWidget = QtWidgets.QWidget(self._central)
        self.contentWidget.setStyleSheet("background: transparent;")
        self.contentLayout = QtWidgets.QHBoxLayout(self.contentWidget)
        self.contentLayout.setContentsMargins(20, 20, 20, 20)
        self.contentLayout.setSpacing(30)

        # 左侧按钮面板
        self.leftPanel = QtWidgets.QWidget(self.contentWidget)
        self.leftPanel.setStyleSheet("background: transparent;")
        self.leftPanel.setObjectName("leftPanel")
        self.leftPanel.setMinimumWidth(160)
        self.leftPanel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.leftLayout = QtWidgets.QVBoxLayout(self.leftPanel)
        self.leftLayout.setContentsMargins(0, 0, 0, 0)
        self.leftLayout.setSpacing(20)
        self.leftLayout.setObjectName("leftLayout")

        self.pushButton = QtWidgets.QPushButton(self.leftPanel)
        self.pushButton.setMinimumSize(QtCore.QSize(120, 80))
        self.pushButton.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.pushButton.setStyleSheet(
            "QPushButton {"
            "    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #2980b9);"
            "    color: white; border: none; border-radius: 15px;"
            "    font-size: 18px; font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5dade2, stop:1 #3498db);"
            "}")
        self.pushButton.setObjectName("pushButton")
        self.leftLayout.addWidget(self.pushButton)

        self.pushButton_2 = QtWidgets.QPushButton(self.leftPanel)
        self.pushButton_2.setMinimumSize(QtCore.QSize(120, 80))
        self.pushButton_2.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.pushButton_2.setStyleSheet(
            "QPushButton {"
            "    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #27ae60, stop:1 #1e8449);"
            "    color: white; border: none; border-radius: 15px;"
            "    font-size: 18px; font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #52be80, stop:1 #27ae60);"
            "}")
        self.pushButton_2.setObjectName("pushButton_2")
        self.leftLayout.addWidget(self.pushButton_2)

        self.pushButton_3 = QtWidgets.QPushButton(self.leftPanel)
        self.pushButton_3.setMinimumSize(QtCore.QSize(120, 50))
        self.pushButton_3.setMaximumHeight(80)
        self.pushButton_3.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.pushButton_3.setStyleSheet(
            "QPushButton {"
            "    background: #bdc3c7; color: #2c3e50; border: none;"
            "    border-radius: 10px; font-size: 14px; font-weight: bold;"
            "}"
            "QPushButton:hover { background: #aab7b8; }")
        self.pushButton_3.setObjectName("pushButton_3")
        self.leftLayout.addWidget(self.pushButton_3)

        self.contentLayout.addWidget(self.leftPanel, 3)

        # 右侧图片面板
        self.rightPanel = QtWidgets.QFrame(self.contentWidget)
        self.rightPanel.setStyleSheet(
            "background: white; border: none; border-radius: 15px;")
        self.rightPanel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.rightPanel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.rightPanel.setObjectName("rightPanel")
        self.rightLayout = QtWidgets.QVBoxLayout(self.rightPanel)
        self.rightLayout.setContentsMargins(15, 15, 15, 15)
        self.rightLayout.setObjectName("rightLayout")

        self.label_2 = QtWidgets.QLabel(self.rightPanel)
        self.label_2.setStyleSheet(
            "border: none; background: #ecf0f1; border-radius: 10px;")
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/img/img/R-C.jpg"))
        self.label_2.setScaledContents(True)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.rightLayout.addWidget(self.label_2)

        self.contentLayout.addWidget(self.rightPanel, 6)
        self.rootLayout.addWidget(self.contentWidget, 1)

        # 底部提示
        self.bottomTip = QtWidgets.QLabel(self._central)
        self.bottomTip.setStyleSheet(
            "color: #95a5a6; font-size: 12px; background: transparent;")
        self.bottomTip.setAlignment(QtCore.Qt.AlignCenter)
        self.bottomTip.setObjectName("bottomTip")
        self.bottomTip.setFixedHeight(40)
        self.rootLayout.addWidget(self.bottomTip)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "泪膜破裂检测系统"))
        self.titleLabel.setText(_translate("Form", "泪膜破裂检测系统"))
        self.subTitle.setText(_translate("Form", "Tear Film Break-up Detection System"))
        self.pushButton.setText(_translate("Form", "图像检测"))
        self.pushButton_2.setText(_translate("Form", "视频检测"))
        self.pushButton_3.setText(_translate("Form", "退出系统"))
        self.bottomTip.setText(_translate("Form", "基于 Placido 环图像的泪膜破裂自动检测"))

import csy_rc
