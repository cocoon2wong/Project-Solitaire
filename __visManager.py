"""
@Author: Conghao Wong
@Date: 2024-11-05 15:48:10
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-05 20:45:09
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os
import tkinter as tk
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageTk

from qpid.args.__args import Args
from qpid.base import BaseManager
from qpid.dataset.agent_based import Agent
from qpid.mods import vis

from .__args import PlaygroundArgs
from .__constant import (DRAW_MODE_PLT, DRAW_MODE_QPID,
                         DRAW_MODE_QPID_PHYSICAL, DRAW_MODES_ALL,
                         MARKER_CIRCLE_RADIUS, MARKER_RADIUS, MARKER_TAG,
                         MAX_HEIGHT, MAX_WIDTH, OBSTACLE_IMAGE_PATH, SEG_MAP_B,
                         SEG_MAP_G, SEG_MAP_R, TEMP_IMG_PATH,
                         TEMP_RGB_IMG_PATH, TEMP_SEG_MAP_PATH)
from .__interface import InterfaceManager


class VisManager(BaseManager):
    def __init__(self, args: Args | None = None, manager: Any = None, name: str | None = None):
        super().__init__(args, manager, name)

        # Args
        self.pg_args = self.args.register_subargs(PlaygroundArgs, 'pg_args')

        # Image containers
        self.image: tk.PhotoImage | None = None
        self.image_shape = None

        self.segmap: ImageTk.PhotoImage | None = None
        self.obstacle_img: ImageTk.PhotoImage | None = None

        # Vis tool
        self.vis_handler = vis.Visualization(manager=self.manager.t,
                                             dataset=self.pg_args.dataset,
                                             clip=self.pg_args.clip)

        # Variables
        self.draw_mode_count = -1
        self.click_count = 0
        self.image_scale = 1.0
        self.image_margin = [0.0, 0.0]

        self.hover_marker_id: int | None = None

        # Init methods
        self.switch_draw_mode()

        # Bind methods on the canvas
        self.canvas.bind("<Motion>", lambda e: self.on_hover_canvas(e))
        self.canvas.bind("<Button-1>", lambda e: self.on_click_canvas(e))

    @property
    def tk_vars(self) -> dict[str, tk.StringVar]:
        return self.manager.tk_vars     # type: ignore

    @property
    def draw_mode(self) -> str:
        return DRAW_MODES_ALL[self.draw_mode_count]

    @property
    def canvas(self):
        return self.manager.get_member(InterfaceManager).canvas

    @property
    def status_label(self):
        return self.manager.get_member(InterfaceManager).var_draw_status

    def switch_draw_mode(self):
        self.draw_mode_count += 1
        self.draw_mode_count %= len(DRAW_MODES_ALL)
        self.status_label.config(text=f'Mode: {self.draw_mode}')

    def draw(self, model_args: Args, agent: Agent):
        m = self.draw_mode
        do = self.vis_handler.draw
        need_resize = False

        if m in [DRAW_MODE_QPID, DRAW_MODE_QPID_PHYSICAL]:
            need_resize = True
            img_save_path = TEMP_RGB_IMG_PATH
            draw_with_plt = False

        elif m == DRAW_MODE_PLT:
            img_save_path = TEMP_IMG_PATH
            draw_with_plt = True

        else:
            raise ValueError(m)

        do(agent=agent,
            frames=[agent.frames[model_args.obs_frames-1]],
            save_name=img_save_path,
            save_name_with_frame=False,
            save_as_images=True,
            draw_with_plt=draw_with_plt)

        if need_resize:
            import cv2
            f = cv2.imread(img_save_path)
            h, w = f.shape[:2]
            if ((h >= MAX_HEIGHT) and (h/w >= MAX_HEIGHT/MAX_WIDTH)):
                self.image_scale = h / MAX_HEIGHT
                self.image_margin = [0, (MAX_WIDTH - w/self.image_scale)//2]
            elif ((w >= MAX_WIDTH) and (h/w <= MAX_HEIGHT/MAX_WIDTH)):
                self.image_scale = w / MAX_WIDTH
                self.image_margin = [(MAX_HEIGHT - h/self.image_scale)//2, 0]
            else:
                raise ValueError

            f = cv2.resize(f, [int(w//self.image_scale),
                               int(h//self.image_scale)])
            _p = os.path.join(os.path.dirname(img_save_path),
                              'resized_' + os.path.basename(img_save_path))
            cv2.imwrite(_p, f)
            img_save_path = _p

        self.image = tk.PhotoImage(file=img_save_path)
        self.canvas.create_image(MAX_WIDTH//2, MAX_HEIGHT//2, image=self.image)

    def draw_segmap(self, segmap: torch.Tensor):
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

        self.segmap = ImageTk.PhotoImage(_segmap)
        self.canvas.create_image(MAX_WIDTH//2, MAX_HEIGHT//2,
                                 image=self.segmap)

    def on_click_canvas(self, event: tk.Event):

        # Do nothing in the plt mode
        if not self.draw_mode in [DRAW_MODE_QPID,
                                  DRAW_MODE_QPID_PHYSICAL]:
            return

        x, y = [event.x, event.y]
        x_ip, y_ip = self.canvas_pixel_to_image_pixel(x, y)
        x_ir, y_ir = self.image_pixel_to_image_real(x_ip, y_ip)

        c = self.click_count
        if c == 0:
            self.clear_markers()
            self.draw_marker(x, y, 'red', text='START')
            self.click_count = 1

            if self.draw_mode == DRAW_MODE_QPID:
                [x_target, y_target] = [x_ir, y_ir]
            elif self.draw_mode == DRAW_MODE_QPID_PHYSICAL:
                [x_target, y_target] = [x_ip, y_ip]

        elif c == 1:
            if self.pg_args.points == 3 and self.draw_mode == DRAW_MODE_QPID:
                self.draw_marker(x, y, 'orange', text='MIDDLE')
                self.click_count = 2
            else:
                self.draw_marker(x, y, 'blue', text='END')
                self.click_count = 0

            if self.draw_mode == DRAW_MODE_QPID:
                [x_target, y_target] = [x_ir, y_ir]

            elif self.draw_mode == DRAW_MODE_QPID_PHYSICAL:
                [x_target, y_target] = [x_ip, y_ip]
                self.tk_vars['px1'].set(str(x_ip))
                self.tk_vars['py1'].set(str(y_ip))
                self.draw_obstacle()

        elif c == 2:
            self.draw_marker(x, y, 'blue', text='END')
            self.click_count = 0

            [x_target, y_target] = [x_ir, y_ir]

        # Save positions
        self.tk_vars[f'px{c}'].set(str(x_target))
        self.tk_vars[f'py{c}'].set(str(y_target))

    def on_hover_canvas(self, event: tk.Event):
        if self.hover_marker_id is not None:
            self.canvas.delete(self.hover_marker_id)

        self.hover_marker_id = self.canvas.create_oval(
            event.x - MARKER_RADIUS,
            event.y - MARKER_RADIUS,
            event.x + MARKER_RADIUS,
            event.y + MARKER_RADIUS,
            fill='white'
        )

    def clear_manual_positions(self):
        self.clear_markers()
        for p in range(self.pg_args.points):
            for i in ['x', 'y']:
                self.tk_vars[f'p{i}{p}'].set('')

    def clear_markers(self):
        self.canvas.delete(MARKER_TAG)

    def draw_marker(self, x: float, y: float,
                    color: str, text: str | None = None):
        if text:
            self.canvas.create_text(x - 2, y - 20 - 2, text=text,
                                    tags=MARKER_TAG, anchor=tk.N, fill='black')
            self.canvas.create_text(x, y - 20, text=text,
                                    tags=MARKER_TAG, anchor=tk.N, fill='white')

        self.canvas.create_oval(x - MARKER_CIRCLE_RADIUS,
                                y - MARKER_CIRCLE_RADIUS,
                                x + MARKER_CIRCLE_RADIUS,
                                y + MARKER_CIRCLE_RADIUS,
                                fill=color, tags=MARKER_TAG)

    def draw_obstacle(self):
        # Get saved positions (image/pixel)
        res = []
        for _i in ['0', '1']:
            for _j in ['px', 'py']:
                _r = self.tk_vars[_j + _i].get()
                if not len(_r):
                    return

                res.append(float(_r))

        # Transform to canvas positions (canvas/pixel)
        x0_cp, y0_cp = self.image_pixel_to_canvas_pixel(*res[:2])
        x1_cp, y1_cp = self.image_pixel_to_canvas_pixel(*res[2:])

        _dx, _dy = (abs(int(x1_cp - x0_cp)), abs(int(y1_cp - y0_cp)))
        img = Image.open(OBSTACLE_IMAGE_PATH).resize((_dx, _dy))
        self.obstacle_img = ImageTk.PhotoImage(img)
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
