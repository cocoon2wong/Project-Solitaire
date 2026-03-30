"""
@Author: Conghao Wong
@Date: 2024-11-05 15:47:04
@LastEditors: Conghao Wong
@LastEditTime: 2026-03-30 10:52:08
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from copy import copy, deepcopy
from typing import Any

import numpy as np
import torch
from PyQt6.QtWidgets import QFileDialog

import qpid
from qpid.args import Args
from qpid.base import BaseManager
from qpid.constant import INPUT_TYPES
from qpid.dataset.agent_based import Agent
from qpid.training import Structure
from qpid.utils import DATASET_DICT, get_mask, move_to_device

from .__args import PlaygroundArgs, args
from .__constant import DRAW_MODE_PLT, DRAW_MODE_QPID, DRAW_MODE_QPID_PHYSICAL
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
        self.args._set_default('load', 'static')
        self.pg_args = self.args.register_subargs(PlaygroundArgs, 'pg_args')

        self.bind_dict = {}

        self.vars: dict[str, Any] = {}

        self.update_var('agent_id', str(self.pg_args.default_agent))
        self.update_var('model_path', self.args.load)

        self.update_var('Dataset', self.args.dataset)
        self.update_var('Split', self.args.split)
        self.update_var('Clip', self.pg_args.clip)

        self.update_var('Dataset_list', [self.args.dataset])
        self.update_var('Split_list', [self.args.split])
        self.update_var('Clip_list', [self.pg_args.clip])

        self.update_var('has_manual_neighbors', False)

        for p in range(self.pg_args.points):
            for i in ['x', 'y']:
                self.vars[f'p{i}{p}'] = None

        # Managers
        self.vis_mgr: VisManager | None = None

        # Variables
        self.t: Structure | None = None

        # Data containers
        self.inputs: list[torch.Tensor] | None = None
        self.outputs: list[torch.Tensor] | None = None
        self.input_and_gt: list[list[torch.Tensor]] | None = None
        self.input_types = None

        # Try to load models from the init args
        self.load(self.vars['model_path'])

        # Interpolation layer
        self.interp_model = None

        # Init dataset-related settings
        self.init_dataset()

    @property
    def agent_index(self) -> int:
        return int(self.vars['agent_id'])

    def bind_var(self, name: str, func):
        self.bind_dict[name] = func

    def update_var(self, name: str, value: Any):
        self.vars[name] = value
        if name in self.bind_dict.keys():
            self.bind_dict[name](value)

    def visit_all_vars(self):
        for key in sorted(self.vars, reverse=True):
            value = self.vars[key]
            self.update_var(key, value)

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

    def choose_weights(self, load=True):
        path = QFileDialog.getExistingDirectory(None, "Choose Weights", "")
        self.update_var('model_path', path)

        if load:
            self.load(path)

    def init_dataset(self):
        self.update_dataset(
            self.vars['Dataset'], split=self.args.split, clip=self.pg_args.clip)
        self.update_var('Dataset_list', list(DATASET_DICT.keys()))

    def update_dataset(self, ds: str, split=None, clip=None):
        self.update_var('Dataset', ds)

        if not split:
            split = list(DATASET_DICT[ds].keys())[0]

        self.update_split(split, clip)
        self.update_var('Split_list', list(DATASET_DICT[ds].keys()))

    def update_split(self, split: str, clip=None):
        self.update_var('Split', split)
        ds = self.vars['Dataset']

        if not clip:
            clip = DATASET_DICT[ds][split][0]

        self.update_var('Clip', clip)
        self.update_var('Clip_list', DATASET_DICT[ds][split])

    def load(self, path: str):
        try:
            terminal_args = args(path)

            self.args._set('dataset', self.vars['Dataset'])
            self.args._set('split', self.vars['Split'])
            self.pg_args._set('clip', self.vars['Clip'])

            # Set datasets
            terminal_args += ['--force_dataset', self.args.dataset,
                              '--force_clip', self.pg_args.clip,
                              '--force_split', self.args.split]

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
                            self.t.args.pred_frames,
                            self.t.args.force_clip)
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
        if not self.vis_mgr or (old_input_types and old_input_types[-1] != self.input_types[-1]):
            self.create_vis_manager()

    def run(self, with_manual_neighbor=False, save_results=True):

        if ((not self.input_and_gt) or
            (not self.vis_mgr) or
            (not self.t) or
                (not len(self.agents))):
            raise ValueError

        # Gather model inputs
        inputs = [i[self.agent_index][None] for i in self.input_and_gt[0]]

        # Read the position of the manual neighbor
        extra_pos = []
        if with_manual_neighbor:
            for _i in ['x', 'y']:
                for _j in range(self.pg_args.points):
                    _v = self.vars[f'p{_i}{_j}']
                    if _v is not None:
                        try:
                            extra_pos.append(_v)
                        except:
                            self.log(
                                f'Illegal position `{_v}`!', level='error')

            if len(extra_pos) == 2 * self.pg_args.points:
                if save_results:
                    self.update_var('has_manual_neighbor', True)
                    self.log('Start running with an addition neighbor' +
                             f'from {extra_pos[0]} to {extra_pos[1]}...')
            else:
                self.update_var('has_manual_neighbor', False)
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

        original_inputs = deepcopy(inputs)

        # Check whether to forecast trajectories for all neighbors
        if self.pg_args.predict_all_neighbors:
            # -> (1, max_nei, obs, dim)
            all_nei = self.t.model.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

            # Filter valid neighbors -> (nei, obs, dim)
            valid_mask = get_mask(torch.abs(all_nei).sum([-1, -2]))
            valid_idx = torch.where(valid_mask.bool())
            current_nei = all_nei[valid_idx]

            obs_idx = self.t.model.input_types.index(INPUT_TYPES.OBSERVED_TRAJ)
            nei_idx = self.t.model.input_types.index(INPUT_TYPES.NEIGHBOR_TRAJ)

            _ego_obs = inputs[obs_idx][0]

            for _nei in current_nei:
                _nei_obs = (_nei + _ego_obs[-1:, :])[None]

                if torch.sum(torch.abs(_nei_obs - _ego_obs)) < 1e-4:
                    continue

                for _idx in range(len(inputs)):
                    if _idx == obs_idx:
                        inputs[_idx] = torch.concat([
                            inputs[_idx],
                            _nei_obs,
                        ], dim=0)

                    elif _idx == nei_idx:
                        inputs[_idx] = torch.concat([
                            inputs[_idx],
                            inputs[_idx][:1] + inputs[obs_idx][:1, None, :, :],
                        ], dim=0)

                    else:
                        inputs[_idx] = torch.concat([
                            inputs[_idx],
                            inputs[_idx][:1],
                        ], dim=0)

        # Forward the model
        with torch.no_grad():
            outputs = self.t.model.implement(inputs, training=None)

        if self.pg_args.predict_all_neighbors:
            # Resort outputs
            outputs[0] = outputs[0][None]

        # Save model inputs/outputs
        self.inputs = original_inputs
        self.outputs = move_to_device(outputs, self.t.device_cpu)

        if not save_results:
            return self.outputs[0]

        # Save results into an `Agent` object
        # Print model outputs
        time = int(1000 * self.t.model.inference_times[-1])
        self.log(f'Running done. Time cost = {time} ms.')

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
        return self.outputs[0]

    def get_random_id(self):
        try:
            n = len(self.agents)
            if self.t:
                n = min(n, self.t.args.batch_size)
            i = np.random.randint(0, n)
            self.update_var('agent_id', str(i))
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
        nei_count = min(nei_count, self.args.max_agents - 1)
        _nei[0, nei_count] = traj - obs.numpy()[0, -1:, :]
        return torch.from_numpy(_nei)


def get_neighbor_count(nei_obs: torch.Tensor | np.ndarray):
    if isinstance(nei_obs, np.ndarray):
        nei_obs = torch.from_numpy(nei_obs)

    nei_mask = get_mask(torch.sum(nei_obs, dim=[-1, -2]))
    return int(torch.sum(nei_mask))
