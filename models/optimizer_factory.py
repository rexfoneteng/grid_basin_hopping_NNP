#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Phan Huu Trong
# @Date:   2025-04-30 14:20:41
# @Email:  phanhuutrong93@gmail.com
# @Last modified by:   vanan
# @Last modified time: 2025-05-16 17:46:15
# @Description: Factory for creating structure optimizers

import logging
from typing import Dict, Any, Optional

from models.optimizer_interface import StructureOptimizer
from models.nnp_optimizer import NNPOptimizer, NNPGaussOptimizer
from models.gamess_optimizer import GamessOptimizer

logger = logging.getLogger(__name__)

class OptimizerFactory:
    """
    Factory class for creating structure optimizers.
    """

    @staticmethod
    def create_optimizer(optimizer_type: str, params: Dict[str, Any]) -> Optional[StructureOptimizer]:
        """
        Create an optimizer of the specified type.
        
        Args:
            optimizer_type: Type of optimizer to create ('nnp' or 'gamess')
            params: Parameters for the optimizer
            
        Returns:
            An instance of the requested optimizer or None if the type is unknown
        """
        if optimizer_type.lower() == "nnp":
            logger.info("Creating NNP Optimizer")
            return NNPOptimizer(params)
        elif optimizer_type.lower() == "dftb3":
            logger.info("Creating DFTB3 Optimizer")
            return GamessOptimizer(params)
        elif optimizer_type.lower() == "nnp_gauss":
            logger.info("Creating NNP-Gauss Optimizer")
            return NNPGaussOptimizer(params)
        else:
            logger.error(f"Unknown optimizer type: {optimizer_type}")
            return None
