# --coding:utf-8--

import sys

import gui
import Breath

sys.modules['gui'].__dict__.clear()
sys.modules['Breath'].__dict__.clear()

gui.py
print('gui')

Breath.py
print('Breath')

