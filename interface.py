"""
@Author: Conghao Wong
@Date: 2025-01-02 20:39:07
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-03 15:36:05
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import logging
import os

from PyQt6.QtWidgets import QMainWindow, QTextEdit

import qpid
from qpid.__root import BaseObject
from qpid.base import BaseManager
from qpid.utils import dir_check

from .__constant import LOG_PATH
from .__playgroundManager import PlaygroundManager
from .window import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow, BaseManager):
    def __init__(self, playground_mgr: PlaygroundManager) -> None:

        super().__init__()
        self.setupUi(self)
        self.p = playground_mgr
        self.p.manager = self

        self.pushButton_run.clicked.connect(self.p.run)
        self.pushButton_random.clicked.connect(self.p.get_random_id)

        self.p.bind_var('agent_id', self.lineEdit_agentid.setText)
        self.lineEdit_agentid.textChanged.connect(
            lambda t: self.p.update_var('agent_id', t))

        self.p.bind_var('model_path', self.label_modelpath.setText)
        self.pushButton_load.clicked.connect(self.p.choose_weights)

        self.comboBox_Dataset.textActivated.connect(
            lambda t: self.p.update_dataset(t))
        self.p.bind_var(
            'Dataset', lambda t: self.comboBox_Dataset.setCurrentText(t))
        self.p.bind_var('Dataset_list', lambda t: (self.comboBox_Dataset.clear(),
                                                   self.comboBox_Dataset.addItems(t)))

        self.comboBox_Split.textActivated.connect(
            lambda t: self.p.update_split(t))
        self.p.bind_var(
            'Split', lambda t: self.comboBox_Split.setCurrentText(t))
        self.p.bind_var('Split_list', lambda t: (self.comboBox_Split.clear(),
                                                 self.comboBox_Split.addItems(t)))

        self.comboBox_Clip.textActivated.connect(
            lambda t: self.p.update_var('Clip', t))
        self.p.bind_var('Clip', lambda t: self.comboBox_Clip.setCurrentText(t))
        self.p.bind_var('Clip_list', lambda t: (self.comboBox_Clip.clear(),
                                                self.comboBox_Clip.addItems(t)))

        self.pushButton_changedataset.clicked.connect(lambda: (self.p.load(self.p.vars['model_path']),
                                                               self.lineEdit_agentid.setText('0')))

        self.pushButton_3.clicked.connect(lambda: print(
            self.p.vars['Dataset'], self.p.vars['Split'], self.p.vars['Clip']))

        self.pushButton_modechange.clicked.connect(lambda e: (self.p.vis_mgr.switch_draw_mode(),
                                                              self.label_mode.setText(self.p.vis_mgr.draw_mode)))

        self.p.visit_all_vars()

        # Redirect all log outputs
        dir_check(os.path.dirname(LOG_PATH))
        qpid.set_log_path(LOG_PATH)
        qpid.set_log_stream_handler(TextboxHandler(self.textEdit_logbar))

        # Redirect logs
        BaseObject.__init__(self.p, name='root')


class TextboxHandler(logging.Handler):
    def __init__(self, box: QTextEdit):
        super().__init__()
        self.box = box

    def emit(self, record):
        msg = self.format(record)
        self.box.append(msg)
