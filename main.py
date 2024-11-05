"""
@Author: Conghao Wong
@Date: 2024-11-05 15:29:32
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-05 20:46:35
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from playground import PlaygroundManager
from qpid.args import Args

if __name__ == '__main__':
    p = PlaygroundManager(Args(sys.argv))
    p.interface_mgr.start_loop()
