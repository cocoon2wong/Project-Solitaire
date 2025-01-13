"""
@Author: Conghao Wong
@Date: 2025-01-06 14:56:32
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-13 17:34:09
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

from PyQt6 import QtCore, QtGui, QtWidgets


class MyButton(QtWidgets.QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.BG_COLOR_NORMAL = QtGui.QColor(236, 236, 236)
        self.BG_COLOR_HOVER = QtGui.QColor(0, 133, 161)
        self.BG_COLOR_PRESS = QtGui.QColor(0, 103, 131)

        self.FONT_COLOR_NORMAL = QtGui.QColor(0, 0, 0)
        self.FONT_COLOR_HOVER = QtGui.QColor(255, 255, 255)
        self.FONT_COLOR_PRESS = QtGui.QColor(255, 255, 255)

        self._font_color = self.FONT_COLOR_NORMAL
        self._bg_color = self.BG_COLOR_NORMAL
        self._updated = 0

        self._bg_anim = QtCore.QVariantAnimation(self)
        self._bg_anim.setDuration(150)
        self._bg_anim.setStartValue(QtGui.QColor(236, 236, 236))
        self._bg_anim.setEndValue(QtGui.QColor(0, 133, 161))
        self._bg_anim.valueChanged.connect(
            lambda color: self.update_color('_bg_color', color))

        self._font_anim = QtCore.QVariantAnimation(self)
        self._font_anim.setDuration(50)
        self._font_anim.setStartValue(QtGui.QColor(0, 0, 0))
        self._font_anim.setEndValue(QtGui.QColor(255, 255, 255))
        self._font_anim.valueChanged.connect(
            lambda color: self.update_color('_font_color', color))

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        self._bg_anim.setStartValue(self._bg_color)
        self._bg_anim.setEndValue(self.BG_COLOR_PRESS)
        self._bg_anim.start()

        self._font_anim.setStartValue(self._font_color)
        self._font_anim.setEndValue(self.FONT_COLOR_PRESS)
        self._font_anim.start()
        return super().mousePressEvent(e)

    def enterEvent(self, e):
        self._bg_anim.setStartValue(self._bg_color)
        self._bg_anim.setEndValue(self.BG_COLOR_HOVER)
        self._bg_anim.start()

        self._font_anim.setStartValue(self._font_color)
        self._font_anim.setEndValue(self.FONT_COLOR_HOVER)
        self._font_anim.start()
        return super().enterEvent(e)

    def leaveEvent(self, e):
        self._bg_anim.setStartValue(self._bg_color)
        self._bg_anim.setEndValue(self.BG_COLOR_NORMAL)
        self._bg_anim.start()

        self._font_anim.setStartValue(self._font_color)
        self._font_anim.setEndValue(self.FONT_COLOR_NORMAL)
        self._font_anim.start()
        return super().leaveEvent(e)

    def update_color(self, name: str, value: QtGui.QColor):
        setattr(self, name, value)

        self.setStyleSheet(f"""
            color: {self._font_color.name()};
            background-color: {self._bg_color.name()};
        """)
