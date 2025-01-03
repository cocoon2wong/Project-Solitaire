"""
@Author: Conghao Wong
@Date: 2024-11-05 15:29:32
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-03 15:32:26
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os
import sys

from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath('.'))

from playground.interface import MainWindow

try:
    import main
except:
    pass

from playground import PlaygroundManager
from qpid.args import Args

if __name__ == '__main__':
    p = PlaygroundManager(Args(sys.argv))
    app = QApplication([])
    main = MainWindow(p)
    main.show()
    sys.exit(app.exec())
