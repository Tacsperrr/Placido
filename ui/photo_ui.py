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

        # 顶部标题
        self.pageTitle = QtWidgets.QLabel(self._central)
        self.pageTitle.setFixedHeight(60)
        self.pageTitle.setStyleSheet(
            "color: #2c3e50; font-size: 22px; font-weight: bold; background: transparent;")
        self.pageTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.pageTitle.setObjectName("pageTitle")
        self.rootLayout.addWidget(self.pageTitle)

        # 中间双栏内容区
        self.contentWidget = QtWidgets.QWidget(self._central)
        self.contentWidget.setStyleSheet("background: transparent;")
        self.contentLayout = QtWidgets.QHBoxLayout(self.contentWidget)
        self.contentLayout.setContentsMargins(40, 10, 40, 10)
        self.contentLayout.setSpacing(30)

        # 左：原始图像
        self.originalFrame = QtWidgets.QFrame(self.contentWidget)
        self.originalFrame.setStyleSheet(
            "background: white; border: none; border-radius: 12px;")
        self.originalFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.originalFrame.setObjectName("originalFrame")
        self.originalLayout = QtWidgets.QVBoxLayout(self.originalFrame)
        self.originalLayout.setContentsMargins(10, 10, 10, 10)
        self.originalLayout.setObjectName("originalLayout")

        self.labelTitle1 = QtWidgets.QLabel(self.originalFrame)
        self.labelTitle1.setMaximumHeight(35)
        self.labelTitle1.setStyleSheet(
            "color: #3498db; font-size: 14px; font-weight: bold;"
            "background: #ebf5fb; border-radius: 8px;")
        self.labelTitle1.setAlignment(QtCore.Qt.AlignCenter)
        self.labelTitle1.setObjectName("labelTitle1")
        self.originalLayout.addWidget(self.labelTitle1)

        self.label = QtWidgets.QLabel(self.originalFrame)
        self.label.setStyleSheet("background: #ecf0f1; border-radius: 8px;")
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.originalLayout.addWidget(self.label)

        self.contentLayout.addWidget(self.originalFrame, 1)

        # 右：检测结果
        self.resultFrame = QtWidgets.QFrame(self.contentWidget)
        self.resultFrame.setStyleSheet(
            "background: white; border: none; border-radius: 12px;")
        self.resultFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.resultFrame.setObjectName("resultFrame")
        self.resultLayout = QtWidgets.QVBoxLayout(self.resultFrame)
        self.resultLayout.setContentsMargins(10, 10, 10, 10)
        self.resultLayout.setObjectName("resultLayout")

        self.labelTitle2 = QtWidgets.QLabel(self.resultFrame)
        self.labelTitle2.setMaximumHeight(35)
        self.labelTitle2.setStyleSheet(
            "color: #27ae60; font-size: 14px; font-weight: bold;"
            "background: #e9f7ef; border-radius: 8px;")
        self.labelTitle2.setAlignment(QtCore.Qt.AlignCenter)
        self.labelTitle2.setObjectName("labelTitle2")
        self.resultLayout.addWidget(self.labelTitle2)

        self.label_2 = QtWidgets.QLabel(self.resultFrame)
        self.label_2.setStyleSheet("background: #ecf0f1; border-radius: 8px;")
        self.label_2.setText("")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.resultLayout.addWidget(self.label_2)

        self.contentLayout.addWidget(self.resultFrame, 1)
        self.rootLayout.addWidget(self.contentWidget, 1)

        # 底部按钮栏
        self.buttonFrame = QtWidgets.QWidget(self._central)
        self.buttonFrame.setFixedHeight(80)
        self.buttonFrame.setStyleSheet("background: transparent;")
        self.buttonFrame.setObjectName("buttonFrame")
        self.buttonLayout = QtWidgets.QHBoxLayout(self.buttonFrame)
        self.buttonLayout.setContentsMargins(50, 0, 50, 0)
        self.buttonLayout.setSpacing(30)
        self.buttonLayout.setObjectName("buttonLayout")

        # 第一个按钮：选择图片
        self.pushButton = QtWidgets.QPushButton(self.buttonFrame)
        self.pushButton.setMinimumSize(QtCore.QSize(140, 45))
        self.pushButton.setStyleSheet(
            "QPushButton { background: #3498db; color: white; border: none;"
            "    border-radius: 22px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #2980b9; }")
        self.pushButton.setObjectName("pushButton")
        self.buttonLayout.addWidget(self.pushButton)

        # 第二个按钮：开始检测
        self.pushButton_2 = QtWidgets.QPushButton(self.buttonFrame)
        self.pushButton_2.setMinimumSize(QtCore.QSize(140, 45))
        self.pushButton_2.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; border: none;"
            "    border-radius: 22px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #1e8449; }")
        self.pushButton_2.setObjectName("pushButton_2")
        self.buttonLayout.addWidget(self.pushButton_2)

        # 第三个按钮：开始检测2（新添加的按钮，与开始检测按钮样式相同）
        self.pushButton_4 = QtWidgets.QPushButton(self.buttonFrame)
        self.pushButton_4.setMinimumSize(QtCore.QSize(140, 45))
        self.pushButton_4.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; border: none;"
            "    border-radius: 22px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #1e8449; }")
        self.pushButton_4.setObjectName("pushButton_4")
        self.buttonLayout.addWidget(self.pushButton_4)

        # 添加弹簧（弹性空间）
        self.buttonLayout.addStretch(1)

        # 第四个按钮：返回
        self.pushButton_3 = QtWidgets.QPushButton(self.buttonFrame)
        self.pushButton_3.setMinimumSize(QtCore.QSize(100, 45))
        self.pushButton_3.setStyleSheet(
            "QPushButton { background: #bdc3c7; color: #2c3e50; border: none;"
            "    border-radius: 22px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #aab7b8; }")
        self.pushButton_3.setObjectName("pushButton_3")
        self.buttonLayout.addWidget(self.pushButton_3)

        self.rootLayout.addWidget(self.buttonFrame)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "图像检测 - 泪膜破裂检测系统"))
        self.pageTitle.setText(_translate("Form", "图像检测"))
        self.labelTitle1.setText(_translate("Form", "原始图像"))
        self.labelTitle2.setText(_translate("Form", "检测结果"))
        self.pushButton.setText(_translate("Form", "选择图片"))
        self.pushButton_2.setText(_translate("Form", "开始检测"))
        self.pushButton_4.setText(_translate("Form", "开始检测2"))  # 新按钮的文本
        self.pushButton_3.setText(_translate("Form", "返回"))