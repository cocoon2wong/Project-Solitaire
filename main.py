"""
@Author: Conghao Wong
@Date: 2024-11-05 15:29:32
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-16 11:28:38
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.abspath('.'))

try:
    import main
except:
    pass

from playground import PlaygroundManager
from qpid.args import Args

if __name__ == '__main__':
    p = PlaygroundManager(Args(sys.argv))
    p.interface_mgr.start_loop()
