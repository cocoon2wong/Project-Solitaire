"""
@Author: Conghao Wong
@Date: 2024-11-05 15:47:04
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-05 20:40:31
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import tkinter as tk
from copy import copy, deepcopy
from tkinter import filedialog
from typing import Any

import numpy as np
import torch

from main import main
from qpid.__root import BaseObject
from qpid.args import Args
from qpid.base import BaseManager
from qpid.constant import INPUT_TYPES
from qpid.dataset.agent_based import Agent
from qpid.training import Structure
from qpid.utils import get_mask, move_to_device

from .__args import PlaygroundArgs, args
from .__constant import DRAW_MODE_PLT, DRAW_MODE_QPID, DRAW_MODE_QPID_PHYSICAL
from .__interface import InterfaceManager
from .__visManager import VisManager


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

            t = main(terminal_args, run_train_or_test=False)
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

    def run(self, with_manual_neighbor=False):

        if not self.input_and_gt or not self.t or not len(self.agents) or not self.vis_mgr:
            raise ValueError

        # Gather model inputs
        inputs = [i[self.agent_index][None] for i in self.input_and_gt[0]]

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
                self.log('Start running with an addition neighbor' +
                         f'from {extra_pos[0]} to {extra_pos[1]}...')
            else:
                extra_pos = []
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
            outputs = self.t.model.implement(inputs, training=None)

        # Save model inputs/outputs
        self.inputs = inputs
        self.outputs = move_to_device(outputs, self.t.device_cpu)

        # Save results into an `Agent` object
        _agent = Agent().load_data(
            deepcopy(self.agents[self.agent_index].zip_data())
        )
        _agent.manager = self.t.agent_manager

        _agent.write_pred(self.outputs[0].numpy()[0])
        _agent.traj_neighbor = self.t.model.get_input(
            self.inputs, INPUT_TYPES.NEIGHBOR_TRAJ).numpy()[0]
        _agent.neighbor_number = get_neighbor_count(_agent.traj_neighbor)

        # Draw results
        self.vis_mgr.draw(self.t.args, _agent)

        # Destory the temp agent
        del _agent

        # Print model outputs
        time = int(1000 * self.t.model.inference_times[-1])
        self.log(f'Running done. Time cost = {time} ms.')

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
            fp = np.array(position).reshape([3, 2]).T
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


def get_neighbor_count(nei_obs: torch.Tensor | np.ndarray):
    if isinstance(nei_obs, np.ndarray):
        nei_obs = torch.from_numpy(nei_obs)

    nei_mask = get_mask(torch.sum(nei_obs, dim=[-1, -2]))
    return int(torch.sum(nei_mask))
