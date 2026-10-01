# -*- coding: utf-8 -*-
"""让测试无论在哪个目录下被调用都能 import 到 `wm`。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
