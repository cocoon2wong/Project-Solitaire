"""
@Author: Conghao Wong
@Date: 2024-11-05 15:48:10
@LastEditors: Conghao Wong
@LastEditTime: 2025-06-17 21:18:00
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Any

import numpy as np
import torch
from PyQt6 import QtCore, QtGui, QtWidgets

from qpid.args.__args import Args
from qpid.base import BaseManager
from qpid.dataset.agent_based import Agent
from qpid.mods import vis

from .__args import PlaygroundArgs
from .__constant import (DRAW_MODE_PLT, DRAW_MODE_QPID,
                         DRAW_MODE_QPID_PHYSICAL, DRAW_MODES_ALL,
                         END_POINT_COLOR, MARKER_RADIUS, MID_POINT_COLOR,
                         OBSTACLE_IMAGE_PATH, SEG_MAP_B, SEG_MAP_G, SEG_MAP_R,
                         START_POINT_COLOR, TEMP_IMG_PATH, TEMP_RGB_IMG_PATH,
                         TEMP_SEG_MAP_PATH)


class VisManager(BaseManager):
    def __init__(self, args: Args | None = None, manager: Any = None, name: str | None = None):
        super().__init__(args, manager, name)

        # Args
        self.pg_args = self.args.register_subargs(PlaygroundArgs, 'pg_args')

        # Image containers
        self.image: QtGui.QImage | None = None

        # self.segmap: ImageTk.PhotoImage | None = None
        # self.obstacle_img: ImageTk.PhotoImage | None = None

        # Vis tool
        self.vis_handler = vis.Visualization(manager=self.manager.t,
                                             dataset=self.args.dataset,
                                             clip=self.pg_args.clip)

        # Variables
        self.draw_mode_count = -1
        self.click_count = 0
        self.image_scale = 1.0
        self.image_margin = [0.0, 0.0]

        self.hover_marker_id: int | None = None

        # Init methods
        self.switch_draw_mode()

        # Set colors and labels of manual points
        n = self.pg_args.points
        if n == 2:
            self.point_colors = [QtGui.QColor(START_POINT_COLOR),
                                 QtGui.QColor(END_POINT_COLOR)]
            self.point_labels = ['START', 'END']
        elif n == 3:
            self.point_colors = [QtGui.QColor(START_POINT_COLOR),
                                 QtGui.QColor(MID_POINT_COLOR),
                                 QtGui.QColor(END_POINT_COLOR)]
            self.point_labels = ['START', 'MIDDLE', 'END']
        else:
            self.log(f'Wrong points `{self.pg_args.points}`',
                     level='error', raiseError=ValueError)

    @property
    def positions(self) -> list[QtCore.QPoint]:
        """
        Positions of all clicked points on the canvas.
        """
        if not 'click' in self.vars.keys():
            self.positions = []
        return self.manager.vars['click']   # type: ignore

    @positions.setter
    def positions(self, positions: list[QtCore.QPoint]):
        self.manager.update_var('click', positions)  # type: ignore

    def append_position(self, pos: QtCore.QPoint):
        self.positions = self.positions + [pos]

    @property
    def vars(self) -> dict[str, Any]:
        """
        The dict of all shared variables.
        """
        return self.manager.vars    # type: ignore

    @property
    def draw_mode(self) -> str:
        return DRAW_MODES_ALL[self.draw_mode_count]

    @property
    def canvas(self) -> QtWidgets.QLabel:
        return self.manager.manager.canvas  # type: ignore

    def switch_draw_mode(self):
        self.draw_mode_count += 1
        self.draw_mode_count %= len(DRAW_MODES_ALL)
        self.manager.update_var('draw_mode', self.draw_mode)  # type: ignore

    def draw(self, model_args: Args, agent: Agent):
        m = self.draw_mode
        do = self.vis_handler.draw

        if m in [DRAW_MODE_QPID, DRAW_MODE_QPID_PHYSICAL]:
            img_save_path = TEMP_RGB_IMG_PATH
            draw_with_plt = False

        elif m == DRAW_MODE_PLT:
            img_save_path = TEMP_IMG_PATH
            draw_with_plt = True

        else:
            raise ValueError(m)

        if self.pg_args.save_full_outputs:
            _dir = os.path.dirname(img_save_path)
            _file = f'_{m}_{self.args.dataset}_{self.pg_args.clip}_{agent.loss_weight}.png'
            img_save_path = os.path.join(_dir, _file)

        do(agent=agent,
           frames=int(agent.frames[model_args.obs_frames-1]),
           save_name=img_save_path,
           save_name_postfix=False,
           draw_with_plt=draw_with_plt)

        self.image = QtGui.QImage(img_save_path)
        self.canvas.update()

    def draw_segmap(self, segmap: torch.Tensor):
        # TODO: This method is now useless
        return

        if self.image is None:
            return

        _segmap = segmap[..., None]
        _segmap_alpha = _segmap
        _segmap = torch.concat([SEG_MAP_R * _segmap,
                                SEG_MAP_G * _segmap,
                                SEG_MAP_B * _segmap,
                                255 * 0.5 * _segmap_alpha], dim=-1)

        _segmap = Image.fromarray(_segmap.numpy().astype(np.uint8))
        _segmap = _segmap.resize((self.image.width(),
                                  self.image.height()))
        _segmap.save(TEMP_SEG_MAP_PATH)
        self.canvas.setPixmap(QPixmap(TEMP_SEG_MAP_PATH))

    def on_click_canvas(self, ev: QtGui.QMouseEvent):

        # Do nothing in the plt mode
        if not self.draw_mode in [DRAW_MODE_QPID,
                                  DRAW_MODE_QPID_PHYSICAL]:
            return

        pos = ev.pos()

        x, y = [pos.x(), pos.y()]
        x_ip, y_ip = self.canvas_pixel_to_image_pixel(x, y)
        x_ir, y_ir = self.image_pixel_to_image_real(x_ip, y_ip)

        c = self.click_count
        if c == 0:
            self.clear_markers()

        self.append_position(pos)

        if c == 0:
            self.click_count = 1
            if self.draw_mode == DRAW_MODE_QPID:
                [x_target, y_target] = [x_ir, y_ir]
            elif self.draw_mode == DRAW_MODE_QPID_PHYSICAL:
                [x_target, y_target] = [x_ip, y_ip]

        elif c == 1:
            if self.pg_args.points == 3 and self.draw_mode == DRAW_MODE_QPID:
                self.click_count = 2
            else:
                self.click_count = 0

            if self.draw_mode == DRAW_MODE_QPID:
                [x_target, y_target] = [x_ir, y_ir]

            elif self.draw_mode == DRAW_MODE_QPID_PHYSICAL:
                [x_target, y_target] = [x_ip, y_ip]
                self.vars['px1'] = x_ip
                self.vars['py1'] = y_ip
                self.draw_obstacle()

        elif c == 2:
            self.click_count = 0

            [x_target, y_target] = [x_ir, y_ir]

        # Save positions
        self.canvas.update()
        self.vars[f'px{c}'] = x_target
        self.vars[f'py{c}'] = y_target

    def on_update_canvas(self, a0: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self.canvas)
        painter.setPen(QtGui.QColor(255, 0, 0))

        # Draw background image
        if self.image:
            w = self.image.width()
            h = self.image.height()

            w_canvas = self.canvas.width() - 2
            h_canvas = self.canvas.height() - 2

            w_delta = w_canvas / w
            h_delta = h_canvas / h

            scale = min(h_delta, w_delta)
            w_scaled = int(w * scale)
            h_scaled = int(h * scale)

            w_margin = (w_canvas - w_scaled) // 2
            h_margin = (h_canvas - h_scaled) // 2

            self.image.scaled(int(w * scale), int(h * scale))

            painter.drawImage(QtCore.QPoint(w_margin, h_margin),
                              self.image.scaled(int(w * scale), int(h * scale)))

            # Update scaling variables
            self.image_scale = 1/scale
            self.image_margin = [h_margin, w_margin]

        # Draw manual points
        if len(self.positions):
            for p, c, t in zip(self.positions,
                               self.point_colors,
                               self.point_labels):
                self.draw_marker(painter, p, c, t)

        painter.end()

    def clear_markers(self):
        self.positions = []
        for p in range(self.pg_args.points):
            for i in ['x', 'y']:
                self.vars[f'p{i}{p}'] = None
        self.canvas.update()

    def draw_marker(self, painter: QtGui.QPainter,
                    pos: QtCore.QPoint,
                    color: QtGui.QColor,
                    text: str | None = None):

        [x, y] = [pos.x(), pos.y()]

        if text:
            # Draw text shadow first
            painter.setPen(QtGui.QColor(0, 0, 0))
            painter.drawText(x-1, y-20-1, text)

            # Draw real text
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.drawText(x, y-20, text)

        painter.setBrush(color)
        painter.drawEllipse(pos, MARKER_RADIUS, MARKER_RADIUS)

    def draw_obstacle(self):
        # Get saved positions (image/pixel)
        # TODO: This method is now useless
        return
        res = []
        for _i in ['0', '1']:
            for _j in ['px', 'py']:
                _r = self.vars[_j + _i]
                if _r is None:
                    return

                res.append(float(_r))

        # Transform to canvas positions (canvas/pixel)
        x0_cp, y0_cp = self.image_pixel_to_canvas_pixel(*res[:2])
        x1_cp, y1_cp = self.image_pixel_to_canvas_pixel(*res[2:])

        _dx, _dy = (abs(int(x1_cp - x0_cp)), abs(int(y1_cp - y0_cp)))
        img = Image.open(OBSTACLE_IMAGE_PATH).resize((_dx, _dy))
        # self.obstacle_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(min(x0_cp, x1_cp) + _dx // 2,
                                 min(y0_cp, y1_cp) + _dy // 2,
                                 image=self.obstacle_img)

    def canvas_pixel_to_image_pixel(self, x: float, y: float) -> tuple[float, float]:
        return (self.image_scale * (y - self.image_margin[0]),
                self.image_scale * (x - self.image_margin[1]))

    def image_pixel_to_image_real(self, x: float, y: float) -> tuple[float, float]:
        return self.vis_handler.pixel2real(np.array([[x, y]]))[0]

    def image_pixel_to_canvas_pixel(self, x: float, y: float) -> tuple[float, float]:
        return (y / self.image_scale + self.image_margin[1],
                x / self.image_scale + self.image_margin[0])
