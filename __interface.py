"""
@Author: Conghao Wong
@Date: 2025-01-02 20:39:07
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-13 20:14:42
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import logging
import os
import sys
from copy import copy

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QMainWindow, QTextEdit

import qpid
from qpid.__root import BaseObject
from qpid.args import Args
from qpid.base import BaseManager
from qpid.utils import dir_check

from .__constant import LOG_PATH
from .__playgroundManager import PlaygroundManager
from .__ui import Ui_Dialog, Ui_MainWindow


class DatasetDialog(QDialog, Ui_Dialog, BaseManager):
    def __init__(self, parent, *args, **kwargs) -> None:
        QDialog.__init__(self, parent, *args, **kwargs)
        BaseManager.__init__(self, manager=parent, name='Dataset Dialog')

        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowStaysOnTopHint)
        self.setupUi(self)
        self.manager: MainWindow

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

        self.pushButton_browsemodel.clicked.connect(
            lambda e: self.manager.p.choose_weights(load=False))

        self.pushButton_ok.clicked.connect(self.on_click_ok)
        self.pushButton_cancel.clicked.connect(self.hide)

    @property
    def p(self) -> PlaygroundManager:
        return self.manager.p

    def show(self) -> None:
        self.old_vars = copy(self.p.vars)
        return super().show()

    def on_click_ok(self):
        vars = self.p.vars
        old_vars = self.old_vars

        if (n := vars['model_path']) != (o := old_vars['model_path']):
            if not len(n):
                self.p.vars['model_path'] = o
            else:
                self.p.load(n)

        if not vars['Clip'] == self.old_vars['Clip']:
            self.manager.change_dataset()

        self.hide()


class MainWindow(QMainWindow, Ui_MainWindow, BaseManager):
    def __init__(self, playground_mgr: PlaygroundManager,
                 app: QApplication) -> None:

        QMainWindow.__init__(self)
        BaseManager.__init__(self)

        self.setupUi(self)
        self.p = playground_mgr
        self.p.manager = self
        self.app = app

        self.dataset_dialog = DatasetDialog(self)
        self.dataset_dialog.hide()

        if len(sys.argv) < 2:
            self.label_bootargs.hide()
            self.label_bootargs_title.hide()
        else:
            self.label_bootargs.setText(' '.join(sys.argv))

        self.pushButton_run.clicked.connect(lambda e: self.p.run(True, True))
        self.pushButton_random.clicked.connect(self.p.get_random_id)

        self.p.bind_var('agent_id', self.lineEdit_agentid.setText)
        self.lineEdit_agentid.textChanged.connect(
            lambda t: self.p.update_var('agent_id', t))

        self.p.bind_var('model_path', lambda path: (
            self.label_modelpath.setText(path),
            self.dataset_dialog.lineEdit_modelpath.setText(path)
        ))
        self.pushButton_load.clicked.connect(
            lambda e: self.p.choose_weights(load=True))

        self.pushButton_dataset.clicked.connect(self.dataset_dialog.show)

        if not self.p.vis_mgr:
            self.p.create_vis_manager()

        self.p.bind_var('draw_mode', lambda t: self.label_mode.setText(t))
        self.pushButton_modechange.clicked.connect(self.v.switch_draw_mode)

        self.canvas.mousePressEvent = self.v.on_click_canvas
        self.canvas.paintEvent = self.v.on_update_canvas

        # Buttons for the manual neighbor
        self.pushButton_clear.hide()
        self.pushButton_clear.clicked.connect(self.v.clear_markers)

        self.pushButton_runwithoutneighbors.hide()
        self.pushButton_runwithoutneighbors.clicked.connect(
            lambda e: self.p.run(with_manual_neighbor=False))

        self.p.bind_var('click', lambda v: (
            (self.pushButton_clear.show(),
             self.pushButton_runwithoutneighbors.show()) if len(v) else
            (self.pushButton_clear.hide(),
             self.pushButton_runwithoutneighbors.hide())
        ))

        self.p.visit_all_vars()

        # Redirect all log outputs
        logger = self.p.logger
        logger.handlers = []

        dir_check(os.path.dirname(LOG_PATH))
        qpid.set_log_path(LOG_PATH)
        qpid.set_log_stream_handler(TextboxHandler(self.textEdit_logbar))
        BaseObject.__init__(self.p, name=self.p.name)

    @property
    def v(self):
        if not self.p.vis_mgr:
            raise ValueError
        return self.p.vis_mgr

    def change_dataset(self):
        app = self.app
        p_new = PlaygroundManager(Args(sys.argv + [
            '--force_dataset', self.p.vars['Dataset'],
            '--force_split', self.p.vars['Split'],
            '--clip', self.p.vars['Clip'],
            '--load', self.p.vars['model_path']
        ]), name='root')

        self.hide()
        self.__init__(p_new, app)
        self.show()


class TextboxHandler(logging.Handler):
    def __init__(self, box: QTextEdit):
        super().__init__()
        self.box = box

    def emit(self, record):
        msg = self.format(record)
        self.box.append(msg)
