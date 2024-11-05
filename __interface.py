"""
@Author: Conghao Wong
@Date: 2024-11-05 15:30:18
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-05 20:44:19
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import logging
import os
import sys
import tkinter as tk
from typing import Any

import qpid
from qpid.args.__args import Args
from qpid.base import BaseManager
from qpid.utils import dir_check

from .__args import PlaygroundArgs
from .__constant import (LOG_PATH, MAX_HEIGHT, MAX_WIDTH, TK_BORDER_WIDTH,
                         TK_TITLE_STYLE)


class InterfaceManager(BaseManager):

    def __init__(self, args: Args | None = None, manager: Any = None, name: str | None = None):
        super().__init__(args, manager, name)

        self.pg_args = self.args.register_subargs(PlaygroundArgs, 'pg_args')

        # Left column
        l_args: dict[str, Any] = {
            # 'background': '#FFFFFF',
            'border': TK_BORDER_WIDTH,
        }

        # Right Column
        r_args: dict[str, Any] = {
            'background': '#FFFFFF',
            'border': TK_BORDER_WIDTH,
        }
        t_args: dict[str, Any] = {
            'foreground': '#000000',
        }

        # Button Frame
        b_args = {
            # 'background': '#FFFFFF',
            # 'border': TK_BORDER_WIDTH,
        }

        # Init base layouts
        self.root.title(self.name)
        self.LF = tk.Frame(self.root, **l_args)
        self.RF = tk.Frame(self.root, **r_args)
        self.BF = tk.Frame(self.root, **b_args)

        self.LF.grid(row=0, column=0, sticky=tk.NW)
        self.RF.grid(row=0, column=1, sticky=tk.NW, rowspan=2)
        self.BF.grid(row=1, column=0, sticky=tk.N)

        # Init the log window (upon the right frame)
        self.LOGF = tk.Frame(self.RF, **r_args)
        self.LOGF.grid(column=0, row=4, columnspan=2)

        self.logbar = tk.Text(self.LOGF, width=89,
                              height=7, **r_args, **t_args)
        self.scroll = tk.Scrollbar(self.LOGF, command=self.logbar.yview)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.logbar.config(yscrollcommand=self.scroll.set)
        self.logbar.pack()

        # Redirect all log outputs
        dir_check(os.path.dirname(LOG_PATH))
        qpid.set_log_path(LOG_PATH)
        qpid.set_log_stream_handler(TextboxHandler(self.logbar))

        # TK components
        # Left frame
        _i = -1

        if not self.pg_args.lite:
            self.text_left_title = tk.Label(
                self.LF, text='Settings', **TK_TITLE_STYLE, **l_args)
            self.text_left_title.grid(
                column=0, row=(_i := _i + 1), sticky=tk.W)

        # Agent ID: ___
        self.text_agent_id_title = tk.Label(
            self.LF, text='Agent ID', **l_args)
        self.text_agent_id_title.grid(
            column=0, row=(_i := _i + 1))

        self.frame_left_id = tk.Frame(
            self.LF, **l_args)
        self.frame_left_id.grid(
            column=0, row=(_i := _i + 1))

        self.box_agent_id = tk.Entry(
            self.frame_left_id, textvariable=self.tk_vars['agent_id'], width=10)
        self.box_agent_id.grid(
            column=0, row=0)

        self.button_agent_id = tk.Button(
            self.frame_left_id, text='Random')
        self.button_agent_id.grid(
            column=1, row=0)

        # Positions of manual neighbors
        self.text_x0 = tk.Label(
            self.LF, text='New Neighbor (x-axis, start)', **l_args)
        self.text_x0.grid(
            column=0, row=(_i := _i + 1))
        self.box_x0 = tk.Entry(
            self.LF, textvariable=self.tk_vars['px0'])
        self.box_x0.grid(
            column=0, row=(_i := _i + 1))

        self.text_y0 = tk.Label(
            self.LF, text='New Neighbor (y-axis, start)', **l_args)
        self.text_y0.grid(
            column=0, row=(_i := _i + 1))
        self.box_y0 = tk.Entry(
            self.LF,  textvariable=self.tk_vars['py0'])
        self.box_y0.grid(
            column=0, row=(_i := _i + 1))

        self.text_x1 = tk.Label(
            self.LF, text='New Neighbor (x-axis, end)', **l_args)
        self.text_x1.grid(
            column=0, row=(_i := _i + 1))
        self.box_x1 = tk.Entry(
            self.LF, textvariable=self.tk_vars['px1'])
        self.box_x1.grid(
            column=0, row=(_i := _i + 1))

        self.text_y1 = tk.Label(
            self.LF, text='New Neighbor (y-axis, end)', **l_args)
        self.text_y1.grid(
            column=0, row=(_i := _i + 1))
        self.box_y1 = tk.Entry(
            self.LF,  textvariable=self.tk_vars['py1'])
        self.box_y1.grid(
            column=0, row=(_i := _i + 1))

        # Right frame
        i_r = -1

        self.text_model_path = tk.Label(
            self.RF, width=60, wraplength=510,
            text=self.tk_vars['model_path'].get(), **r_args, **t_args)

        if not self.pg_args.lite:
            self.text_right_title = tk.Label(
                self.RF, text='Predictions',
                **TK_TITLE_STYLE, **r_args, **t_args)
            self.text_right_title.grid(
                column=0, row=(i_r := i_r + 1), sticky=tk.W)

            self.text_model_path_title = tk.Label(
                self.RF, text='Model Path:', width=16, anchor=tk.E, **r_args, **t_args)
            self.text_model_path_title.grid(
                column=0, row=(i_r := i_r + 1))
            self.text_model_path.grid(column=1, row=i_r)

        self.canvas = tk.Canvas(self.RF, width=MAX_WIDTH,
                                height=MAX_HEIGHT, **r_args)
        self.canvas.grid(column=0, row=(i_r := i_r + 1), columnspan=2)

        # Bottom frame
        _i = 10

        self.button_run = tk.Button(
            self.BF, text='Run Prediction', **b_args)
        self.button_run.grid(
            column=0, row=(_i := _i + 1), sticky=tk.N)

        self.button_run_original = tk.Button(
            self.BF, text='Run Prediction (Original)', **b_args)
        self.button_run_original.grid(
            column=0, row=(_i := _i + 1), sticky=tk.N)

        self.button_load_model = tk.Button(
            self.BF, text='Load Model Weights', **b_args)
        self.button_load_model.grid(
            column=0, row=(_i := _i + 1), sticky=tk.N)

        self.button_clear_manual_inputs = tk.Button(
            self.BF, text='Clear Manual Inputs', **b_args)
        self.button_clear_manual_inputs.grid(
            column=0, row=(_i := _i + 1), sticky=tk.N)

        self.button_switch_draw_mode = tk.Button(
            self.BF, text='Switch Mode', **b_args)
        self.button_switch_draw_mode.grid(
            column=0, row=(_i := _i + 1), sticky=tk.N)

        self.var_draw_status = tk.Label(self.BF, text='Mode: _', **l_args)
        self.var_draw_status.grid(
            column=0, row=(_i := _i + 1))

        if len(sys.argv) > 1:
            self.text_args = tk.Label(
                self.BF, text=f'Args: {sys.argv[1:]}', wraplength=188, **l_args)
            self.text_args.grid(
                column=0, row=(_i := _i + 1))

    def start_loop(self):
        self.root.mainloop()

    @property
    def tk_vars(self) -> dict[str, tk.StringVar]:
        return self.manager.tk_vars     # type: ignore

    @property
    def root(self) -> tk.Tk:
        return self.manager.root    # type: ignore


class TextboxHandler(logging.Handler):
    def __init__(self, box: tk.Text):
        super().__init__()
        self.box = box

    def emit(self, record) -> None:
        msg = self.format(record)
        self.box.insert(tk.END, msg + '\n')
        self.box.yview_moveto(1.0)
