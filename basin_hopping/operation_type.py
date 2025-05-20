#!/usr/bin/env python3

from enum import Enum

class OperationType(Enum):
    """Enum for the type of structure generation operation"""
    FLIP = "flip"
    ATTACH_ROTATE = "attach_rotate"
    ADD_PROTON = "add_proton"
    FUNCTIONAL_TO_CH = "functional_to_ch"
    CH_TO_METHYL = "ch_to_methyl"