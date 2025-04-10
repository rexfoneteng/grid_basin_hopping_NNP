import os
import logging
import torch
import schnetpack as spk


logger = logging.getLogger(__name__)

class NNPModelLoader:
    """Helper class for loading NNP"""

    @staticmethod
    def load_model(model_params):
        """Load NNP"""
        try:
            # Get model parameters
            state_dict_path = model_params.get("state_dict")
            prop_stats_path = model_params.get("prop_stats")
            device = model_params.get("device", "cpu")

            if not state_dict_path or not os.path.exists(state_dict_path):
                logger.error(f"Model state_dict file not found: {state_dict_path}")
                return None
            
            # Load model
            state_dict_basename = os.path.basename(state_dict_path)
            
            if state_dict_basename.endswith(".pth.tar"):
                # Load property statistics
                if not prop_stats_path or not os.path.exists(prop_stats_path):
                    logger.error(f"Property stats file not found: {prop_stats_path}")
                    return None
                
                prop_stats = torch.load(prop_stats_path)
                means, stddevs = prop_stats["means"], prop_stats["stddevs"]
                
                # Define SchNet model
                in_module_params = model_params.get("in_module", {
                    "n_atom_basis": 128,
                    "n_filters": 128,
                    "n_gaussians": 15,
                    "charged_systems": True,
                    "n_interactions": 4,
                    "cutoff": 15.0
                })
                
                in_module = spk.representation.SchNet(**in_module_params)
                
                out_module = spk.atomistic.Atomwise(
                    n_in=in_module.n_atom_basis,
                    property="energy",
                    mean=means["energy"],
                    stddev=stddevs["energy"],
                    derivative="force",
                    negative_dr=True
                )
                
                model = spk.AtomisticModel(
                    representation=in_module,
                    output_modules=out_module
                )
                
                # Load state dictionary
                state_dict = torch.load(state_dict_path, map_location=device)
                model.load_state_dict(state_dict["model"])
                
            elif state_dict_basename == "best_model":
                model = torch.load(state_dict_path, map_location=device)
                
            else:
                raise ValueError(f"Unknown model format: {state_dict_basename}")
                
            # Set model to evaluation mode
            model.eval()
            logger.info("NNP model loaded successfully")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading NNP model: {str(e)}", exc_info=True)
            return None

