#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/14
"""This module contains several Deep Learning model for Modeling sentence pairs


All model classes must inherit class `BaseModel` (defined in model.py) 
    and implement abstract method `_build_logits`.
"""

# import selected Classes into the package level so they can be convieniently imported from the package.
# use from model import TextCNN instead of from model.base_model import TextCNN
from bcnn import BCNN
from abcnn import ABCNN
from abrnn import ABRNN

# from model import *
__all__ = ["BCNN", "ABCNN", "ABRNN"]

