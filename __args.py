"""
@Author: Conghao Wong
@Date: 2024-11-05 15:39:57
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-11 20:41:30
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import sys

from qpid.args import TEMPORARY, EmptyArgs


class PlaygroundArgs(EmptyArgs):

    @property
    def draw_seg_map(self) -> int:
        """
        Choose whether to draw segmentation maps on the canvas.
        """
        return self._arg('draw_seg_map', 1, TEMPORARY)

    @property
    def points(self) -> int:
        """
        The number of points to simulate the trajectory of manual
        neighbor. It only accepts `2` or `3`.
        """
        return self._arg('points', 2, TEMPORARY)

    @property
    def lite(self) -> int:
        """
        Choose whether to show the lite version of tk window.
        """
        return self._arg('lite', 0, TEMPORARY)

    @property
    def physical_manual_neighbor_mode(self) -> float:
        """
        Mode for the manual neighbor on segmentation maps.
        - Mode `1`: Add obstacles to the given position;
        - Mode `0`: Set areas to be walkable.
        """
        return self._arg('physical_manual_neighbor_mode', 1.0, TEMPORARY)

    @property
    def weight(self) -> str:
        """
        The default weights to load.
        """
        return self._arg('weight', 'static', TEMPORARY, short_name='w')

    @property
    def dataset(self) -> str:
        """
        The dataset to run this playground.
        It accepts `'ETH-UCY'`, `'SDD'`, `'NBA'`, or `'nuScenes_ov'`.
        """
        return self._arg('dataset', 'ETH-UCY', argtype=TEMPORARY)

    @property
    def clip(self) -> str:
        """
        The video clip to run this playground.
        """
        return self._arg('clip', 'zara1', argtype=TEMPORARY)

    @property
    def do_not_draw_neighbors(self) -> int:
        """
        Choose whether to draw neighboring-agents' trajectories.
        """
        return self._arg('do_not_draw_neighbors', 0, argtype=TEMPORARY)

    @property
    def save_full_outputs(self) -> int:
        """
        Choose whether to save all outputs as images.
        """
        return self._arg('save_full_outputs', 0, argtype=TEMPORARY)

    @property
    def compute_social_diff(self) -> int:
        return self._arg('compute_social_diff', 0, argtype=TEMPORARY)


def args(model_path: str):
    return ['main.py',
            '--model', 'MKII',
            '--loads', f'{model_path},speed',
            '-bs', '4000',
            '--test_mode', 'one',
            '--draw_full_neighbors', '1'] + sys.argv
