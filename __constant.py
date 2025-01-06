"""
@Author: Conghao Wong
@Date: 2024-11-05 17:10:05
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-06 18:13:52
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from qpid.utils import get_relative_path

MAX_HEIGHT = 480
MAX_WIDTH = 640

DRAW_MODE_PLT = 'PLT'
DRAW_MODE_QPID = 'Manual Neighbor'
DRAW_MODE_QPID_PHYSICAL = 'Interactive (PC)'

START_POINT_COLOR = '#FF0000'
MID_POINT_COLOR = '#FFFF00'
END_POINT_COLOR = '#0000FF'

LOG_PATH = './temp_files/playground/run.log'
TK_BORDER_WIDTH = 5
TK_TITLE_STYLE = dict(font=('', 24, 'bold'),
                      height=2)

# DRAW_MODES_ALL = [DRAW_MODE_QPID, DRAW_MODE_PLT, DRAW_MODE_QPID_PHYSICAL]
DRAW_MODES_ALL = [DRAW_MODE_QPID, DRAW_MODE_PLT]

TEMP_IMG_PATH = './temp_files/playground/fig.png'
TEMP_SEG_MAP_PATH = './temp_files/playground/seg.png'
TEMP_RGB_IMG_PATH = './temp_files/playground/fig_rgb.png'

OBSTACLE_IMAGE_PATH = get_relative_path(__file__, 'obstacle.png')

MAX_HEIGHT = 480
MAX_WIDTH = 640

MARKER_TAG = 'indicator'
MARKER_CIRCLE_RADIUS = 3
MARKER_RADIUS = 5

SEG_MAP_R = 0xff
SEG_MAP_G = 0xa3
SEG_MAP_B = 0x7f
