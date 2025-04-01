"""
Basin hopping module for structure optimization.

This module provides functionality for performing basin hopping optimization
on molecular structures using neural network potentials.
"""

from basin_hopping.operation_type import OperationType
from basin_hopping.basin_hopping_generator import BasinHoppingGenerator

__all__ = ['OperationType', 'BasinHoppingGenerator']
