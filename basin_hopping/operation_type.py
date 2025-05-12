# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-04-11 12:23:17
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-04-14 11:30:32
# @Description:

from enum import Enum

class OperationType(Enum):
    """Enum for the type of structure generation operation"""
    FLIP = "flip"
    ATTACH_ROTATE = "attach_rotate"
    ADD_PROTON = "add_proton"