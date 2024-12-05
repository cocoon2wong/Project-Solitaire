"""
@Author: Conghao Wong
@Date: 2024-11-05 15:47:04
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-05 15:08:49
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os
import tkinter as tk
from copy import copy, deepcopy
from tkinter import filedialog
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import qpid
from qpid.__root import BaseObject
from qpid.args import Args
from qpid.base import BaseManager
from qpid.constant import INPUT_TYPES
from qpid.dataset.agent_based import Agent
from qpid.training import Structure
from qpid.utils import dir_check, get_mask, move_to_device

from .__args import PlaygroundArgs, args
from .__constant import (DRAW_MODE_PLT, DRAW_MODE_QPID,
                         DRAW_MODE_QPID_PHYSICAL, LOG_PATH)
from .__interface import InterfaceManager
from .__visManager import VisManager

# Configs for computing and drawing the social matrix
COLOR_HIGH = np.array([0xf9, 0xcf, 0x62])
COLOR_LOW = np.array([0x74, 0x8b, 0xe2])

RADIUS = 2.0
HALF_POINTS = 6


class PlaygroundManager(BaseManager):
    def __init__(self, args: Args | None = None, manager: Any = None, name: str | None = None):
        super().__init__(args, manager, name)

        # Args
        self.pg_args = self.args.register_subargs(PlaygroundArgs, 'pg_args')

        # TK Vars
        self.root = tk.Tk()
        self.tk_vars: dict[str, tk.StringVar] = {}
        self.tk_vars['agent_id'] = tk.StringVar(value='0')
        self.tk_vars['model_path'] = tk.StringVar(value=self.pg_args.weight)

        for p in range(self.pg_args.points):
            for i in ['x', 'y']:
                self.tk_vars[f'p{i}{p}'] = tk.StringVar()

        # Managers
        self.interface_mgr = InterfaceManager(manager=self, name='Playground')
        BaseObject.__init__(self, name='root')      # Redirect logs
        self.vis_mgr: VisManager | None = None

        # Variables
        self.t: Structure | None = None

        # Data containers
        self.inputs: list[torch.Tensor] | None = None
        self.outputs: list[torch.Tensor] | None = None
        self.input_and_gt: list[list[torch.Tensor]] | None = None
        self.input_types = None

        # Try to load models from the init args
        self.load(self.model_path)

        # Interpolation layer
        self.interp_model = None

        # Bind methods to the TK canvas
        self.interface_mgr.button_agent_id.bind(
            "<Button-1>", lambda e: self.get_random_id())
        self.interface_mgr.button_run.bind(
            "<Button-1>", lambda e: self.run(with_manual_neighbor=True))
        self.interface_mgr.button_run_original.bind(
            "<Button-1>", lambda e: self.run(with_manual_neighbor=False))
        self.interface_mgr.button_load_model.bind(
            "<Button-1>", lambda e: self.choose_weights())

        if self.pg_args.compute_social_diff:
            self.interface_mgr.button_load_model["text"] = 'Compute Social Heatmap'
            self.interface_mgr.button_load_model.bind(
                "<Button-1>", lambda e: self.compute_social_matrix())

            self.interface_mgr.button_clear_manual_inputs["text"] = 'Show Social Heatmap'
            self.interface_mgr.button_clear_manual_inputs.bind(
                "<Button-1>", lambda e: self.show_social_matrix())

    @property
    def agent_index(self) -> int:
        return int(self.tk_vars['agent_id'].get())

    @property
    def model_path(self) -> str:
        return self.tk_vars['model_path'].get()

    @property
    def agents(self):
        if self.t:
            agents = self.t.agent_manager.agents
            if len(agents):
                pass
            else:
                self.log('No Agent Data!', level='error')
                raise ValueError
        else:
            self.log('No Model Loaded!', level='error')
            raise ValueError
        return agents

    def create_vis_manager(self):
        self.vis_mgr = VisManager(manager=self)

        # Bind button events
        self.interface_mgr.button_switch_draw_mode.bind(
            "<Button-1>", lambda e: self.vis_mgr.switch_draw_mode())  # type: ignore
        self.interface_mgr.button_clear_manual_inputs.bind(
            "<Button-1>", lambda e: self.vis_mgr.clear_manual_positions())  # type: ignore

    def choose_weights(self):
        path = filedialog.askdirectory(initialdir='./')
        self.interface_mgr.text_model_path.config(text=path)
        self.load(path)

    def load(self, path: str):
        try:
            terminal_args = args(path)

            match self.pg_args.dataset:
                case 'ETH-UCY':
                    _split = 'zara1'
                case 'SDD':
                    _split = 'sdd'
                case 'NBA':
                    _split = 'nba50k'
                case 'nuScenes_ov':
                    _split = 'nuScenes_ov_v1.0'
                    # self.t.args._set('interval', 0.5)
                case _:
                    raise ValueError('Wrong dataset settings!')

            # Set datasets
            terminal_args += ['--force_dataset', self.pg_args.dataset,
                              '--force_clip', self.pg_args.clip,
                              '--force_split', _split]

            t = qpid.entrance(terminal_args, train_or_test=False)
            self.t = t

            # Init models and datasets
            self.init_model_and_data()
            self.log(
                f'Model `{t.model.name}` and dataset ({self.pg_args.clip}) loaded.')

        except Exception as e:
            self.log(
                f'An error occurred during loading the model. Details = {e}.')

    def init_model_and_data(self):
        if not self.t:
            self.log('Model NOT loaded!', level='error')
            raise ValueError

        # Create model(s)
        self.t.create_model()

        # Check input types
        if not INPUT_TYPES.NEIGHBOR_TRAJ in self.t.model.input_types:
            self.t.model.input_types.append(INPUT_TYPES.NEIGHBOR_TRAJ)

        old_input_types = self.input_types
        self.input_types = (self.t.model.input_types,
                            self.t.args.obs_frames,
                            self.t.args.pred_frames)
        self.t.agent_manager.set_types(self.t.model.input_types,
                                       self.t.model.label_types)

        # Load dataset files
        if ((self.input_and_gt is None) or
                (self.input_types != old_input_types)):
            self.log('Reloading dataset files...')
            ds = self.t.agent_manager.clean().make(self.t.args.force_clip, training=False)
            self._agents = self.t.agent_manager.agents
            self.input_and_gt = list(ds)[0]

        # Create vis manager
        if not self.vis_mgr:
            self.create_vis_manager()

    def run(self, with_manual_neighbor=False, save_results=True):

        if not self.input_and_gt or not self.t or not len(self.agents) or not self.vis_mgr:
            raise ValueError

        # Gather model inputs
        inputs = [i[self.agent_index][None] for i in self.input_and_gt[0]]
        inputs_original = deepcopy(inputs)

        # Read the position of the manual neighbor
        extra_pos = []
        if with_manual_neighbor:
            for _i in ['x', 'y']:
                for _j in range(self.pg_args.points):
                    _v = self.tk_vars[f'p{_i}{_j}'].get()
                    if len(_v):
                        try:
                            extra_pos.append(float(_v))
                        except:
                            self.log(
                                f'Illegal position `{_v}`!', level='error')

            if len(extra_pos) == 2 * self.pg_args.points:
                if save_results:
                    self.log('Start running with an addition neighbor' +
                             f'from {extra_pos[0]} to {extra_pos[1]}...')
            else:
                extra_pos = []
                with_manual_neighbor = False
                if save_results:
                    self.log('Start running...')

        # Prepare manual neighbors (if needed)
        if len(m := extra_pos):
            if self.vis_mgr.draw_mode in [DRAW_MODE_PLT, DRAW_MODE_QPID]:
                nei = self.add_one_neighbor(inputs, m)
                _i = self.t.model.input_types.index(INPUT_TYPES.NEIGHBOR_TRAJ)
                inputs[_i] = nei

            elif self.vis_mgr.draw_mode in [DRAW_MODE_QPID_PHYSICAL]:
                # TODO: PC MODE
                self.log('The PC Mode is not available now, please try other modes.',
                         level='warning')
                raise NotImplementedError

        # Forward the model
        with torch.no_grad():

            # Compute the social diff value
            if self.pg_args.compute_social_diff and with_manual_neighbor:
                repeats = 100

                for _ii, _item in enumerate(inputs_original):
                    inputs_original[_ii] = torch.repeat_interleave(
                        _item, repeats, dim=0)
                outputs_original = self.t.model.implement(
                    inputs_original, training=None)
                outputs_original[0] = torch.mean(
                    outputs_original[0], dim=0, keepdim=True)

                for _ii, _item in enumerate(inputs):
                    inputs[_ii] = torch.repeat_interleave(
                        _item, repeats, dim=0)
                outputs = self.t.model.implement(inputs, training=None)
                outputs[0] = torch.mean(outputs[0], dim=0, keepdim=True)

                max_mod = torch.abs(outputs[0] - outputs_original[0])
                max_mod = torch.mean(max_mod, dim=[0, 1])
                max_mod = torch.max(max_mod)

                x_current = self.tk_vars[f"px{self.pg_args.points-1}"].get()
                y_current = self.tk_vars[f"py{self.pg_args.points-1}"].get()

                if save_results:
                    self.log(
                        f'Max socially modification: `{max_mod}`, neighbor at `{x_current}`, `{y_current}`.')
                else:
                    return x_current, y_current, max_mod

            else:
                outputs = self.t.model.implement(inputs, training=None)

        # Save model inputs/outputs
        self.inputs = inputs
        self.outputs = move_to_device(outputs, self.t.device_cpu)

        # Print model outputs
        time = int(1000 * self.t.model.inference_times[-1])
        self.log(f'Running done. Time cost = {time} ms.')

        if not save_results:
            return None

        # Save results into an `Agent` object
        _agent = Agent().load_data(
            deepcopy(self.agents[self.agent_index].zip_data())
        )
        _agent.manager = self.t.agent_manager

        _agent.write_pred(self.outputs[0].numpy()[0])

        if self.pg_args.do_not_draw_neighbors:
            _agent.traj_neighbor = np.zeros_like(_agent.traj_neighbor)
            _agent.neighbor_number = 1

        else:
            _agent.traj_neighbor = self.t.model.get_input(
                self.inputs, INPUT_TYPES.NEIGHBOR_TRAJ).numpy()[0]
            _agent.neighbor_number = get_neighbor_count(_agent.traj_neighbor)

        # Store agent index (this variable will be not used when visualizing)
        _agent.loss_weight = self.agent_index

        # Draw results
        self.vis_mgr.draw(self.t.args, _agent)

        # Destory the temp agent
        del _agent
        return None

    def get_random_id(self):
        try:
            n = len(self.agents)
            if self.t:
                n = min(n, self.t.args.batch_size)
            i = np.random.randint(0, n)
            self.tk_vars['agent_id'].set(str(i))
        except:
            pass

    def add_one_neighbor(self, inputs: list[torch.Tensor],
                         position: list[float]):

        if not self.t:
            raise ValueError

        obs = self.t.model.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        nei = self.t.model.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        _nei: np.ndarray = copy(nei.numpy())
        steps = _nei.shape[-2]

        # Interpolate the new neighbor's trajectory
        if len(position) == 4:
            xp = np.array([0, steps-1])
            fp = np.array(position).reshape([2, 2]).T
            x = np.arange(steps)
            traj = np.column_stack([np.interp(x, xp, fp[:, 0]),
                                    np.interp(x, xp, fp[:, 1])])

        elif len(position) == 6:
            xp = np.array([0, steps//2, steps-1])
            fp = np.array(position).reshape([2, 3]).T
            x = np.arange(steps)

            from qpid.model.layers.interpolation import \
                LinearSpeedInterpolation
            if self.interp_model is None:
                self.interp_model = LinearSpeedInterpolation()

            traj = self.interp_model.forward(
                index=torch.tensor(xp),
                value=torch.tensor(fp),
                init_speed=torch.tensor((fp[2:] - fp[:1])/steps)
            ).numpy()
            traj = np.concatenate([fp[:1], traj], axis=0)

        else:
            raise ValueError(len(position))

        nei_count = get_neighbor_count(_nei)
        _nei[0, nei_count] = traj - obs.numpy()[0, -1:, :]
        return torch.from_numpy(_nei)

    def compute_social_matrix(self, delta=RADIUS,
                              x_delta=HALF_POINTS, y_delta=HALF_POINTS):

        tk_vars_copy = {}
        done_list = []
        for _key, _value in self.tk_vars.items():
            if _key.startswith('p') and len(_value.get()):
                tk_vars_copy[_key] = float(_value.get())

        results = []
        for _x in range(-x_delta, x_delta):
            for _j in range(self.pg_args.points):
                _v = tk_vars_copy[f'px{_j}']
                self.tk_vars[f'px{_j}'].set(str((__x := _v + delta * _x)))

            for _y in range(-y_delta, y_delta):
                for _j in range(self.pg_args.points):
                    _v = tk_vars_copy[f'py{_j}']
                    self.tk_vars[f'py{_j}'].set(str((__y := _v + delta * _y)))

                    if not (p := (int(__x*10000)/10000, int(__y*10000)/10000)) in done_list:
                        done_list.append(p)
                    else:
                        continue

                    v = self.run(with_manual_neighbor=True, save_results=False)
                    results.append(
                        [float(v[0]), float(v[1]), float(v[2].numpy())])

        save_path = os.path.join(
            dir_check(os.path.dirname(LOG_PATH)), 'social_matrix.txt')
        np.savetxt(save_path, np.array(results))
        self.log(f'Social matrix saved at `{save_path}`.')
        self.show_social_matrix()

    def show_social_matrix(self):
        data_path = os.path.join(
            dir_check(os.path.dirname(LOG_PATH)), 'social_matrix.txt')

        data = np.loadtxt(data_path)

        plt.close('Social Matrix')
        plt.figure('Social Matrix')

        v_min = data.T[-1].min()
        v_max = data.T[-1].max()

        for _x, _y, _v in data:

            _radius = RADIUS
            _color = COLOR_LOW + (COLOR_HIGH - COLOR_LOW) * \
                ((_v - v_min)/(v_max - v_min))
            _pos = (_x, _y)

            _circle = plt.Circle(_pos, _radius,
                                 fill=True, color=list(_color/255),
                                 alpha=0.6)
            plt.gca().add_artist(_circle)
            plt.plot(_x, _y)

        plt.axis('equal')
        plt.show()


def get_neighbor_count(nei_obs: torch.Tensor | np.ndarray):
    if isinstance(nei_obs, np.ndarray):
        nei_obs = torch.from_numpy(nei_obs)

    nei_mask = get_mask(torch.sum(nei_obs, dim=[-1, -2]))
    return int(torch.sum(nei_mask))
