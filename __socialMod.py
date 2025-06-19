"""
@Author: Conghao Wong
@Date: 2025-06-18 19:18:10
@LastEditors: Conghao Wong
@LastEditTime: 2025-06-19 16:09:15
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from qpid.base import BaseManager
from qpid.utils import dir_check

from .__constant import LOG_PATH

COLOR_HIGH = "#ffdc85"
COLOR_LOW = "#7a91de"

RADIUS = 2.0
HALF_POINTS = 6


def compute_social_mod(manager: BaseManager,
                       vars: dict[str, Any],
                       run_func: Callable[[bool], torch.Tensor],
                       delta_pos: float = RADIUS,
                       x_half_points: int = HALF_POINTS,
                       y_half_points: int = HALF_POINTS,
                       repeats: int = 10):

    save_path = os.path.join(dir_check(os.path.dirname(LOG_PATH)),
                             'social_modification_matrix.txt')

    # Save current position of the manual neighbor as the center point
    vars_backup: dict[str, Any] = {}
    keys = [k for k in vars.keys() if k.startswith(('px', 'py'))]
    for key in keys:
        vars_backup[key] = vars[key]

    if None in vars_backup.values():
        manager.log('Failed to compute social modifications. ' +
                    'Please check if the manual neighbor is properly set.' +
                    '(Click on the canvas to set start/end points.)',
                    level='error')
        return

    # Start sample \delta x and \delta y
    points = len(keys) // 2
    results: list[tuple[float, float, float]] = []
    done_list: list[tuple[int, int]] = []

    for dx in range(-x_half_points, x_half_points):
        # Re-assign manual-neighbor's position
        for p in range(points):
            vars[f'px{p}'] = vars_backup[f'px{p}'] + delta_pos * dx

        for dy in range(-y_half_points, y_half_points):
            for p in range(points):
                vars[f'py{p}'] = vars_backup[f'py{p}'] + delta_pos * dy

            # Prevent from computing multiple times
            if (flag := (dx, dy)) in done_list:
                continue
            else:
                done_list.append(flag)

            # Run original predictions (without manual neighbor)
            # (Save the mean prediction over multiple times)
            r = [run_func(False) for _ in range(repeats)]
            r = torch.mean(torch.stack(r), dim=0)

            # Do intervention (with manual neighbor)
            r_i = [run_func(True) for _ in range(repeats)]
            r_i = torch.mean(torch.stack(r_i), dim=0)

            max_mod = float(torch.max(torch.abs(r - r_i)))
            results.append((vars[f'px{points - 1}'],
                            vars[f'py{points - 1}'],
                            max_mod))

    # Discard all changes
    for k in vars_backup.keys():
        vars[k] = vars_backup[k]

    # Save modification matrix and visualize
    save_path = os.path.join(dir_check(os.path.dirname(LOG_PATH)),
                             'social_modification_matrix.txt')
    np.savetxt(save_path, np.array(results))
    manager.log(f'Social modification matrix saved at `{save_path}`.')

    visualize_social_modification(save_path)


def visualize_social_modification(file_path: str, smooth_size=5):
    from scipy.interpolate import interp2d

    data = np.loadtxt(file_path)
    x: np.ndarray = np.array(list(set(data.T[0])))
    y: np.ndarray = np.array(list(set(data.T[1])))
    v: np.ndarray = data.T[2]

    x.sort()
    y.sort()

    interp = interp2d(x, y, v, 'linear')

    dx = (x.max() - x.min())/(len(x) * smooth_size)
    dy = (y.max() - y.min())/(len(y) * smooth_size)
    x_interp = np.arange(x.min(), x.max(), dx)
    y_interp = np.arange(y.min(), y.max(), dy)
    v_interp = interp(x_interp, y_interp)

    X_interp, Y_interp = np.meshgrid(x_interp, y_interp)

    plt.close('Social Matrix')
    plt.figure('Social Matrix')

    cmap = LinearSegmentedColormap.from_list('diff', [COLOR_LOW, COLOR_HIGH])
    plt.contourf(X_interp, Y_interp, v_interp, levels=100, cmap=cmap)

    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.colorbar()
    plt.show()
