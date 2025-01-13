"""
@Author: Conghao Wong
@Date: 2024-11-05 15:29:32
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-13 19:09:45
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os
import sys

from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath('.'))

try:
    import main
except:
    pass

from playground import MainWindow, PlaygroundManager
from qpid.args import Args

if __name__ == '__main__':
    p = PlaygroundManager(Args(sys.argv), name='root')
    app = QApplication([])
    main = MainWindow(p, app)
    main.show()
    sys.exit(app.exec())
