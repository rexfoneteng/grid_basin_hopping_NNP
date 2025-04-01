from ase.optimize import LBFGS
from schnetpack.interfaces.ase_interface import AseInterface
import os

class CustomInterface(AseInterface):
    """
    Extension of AseInterface that uses LBFGS optimizer for geometry optimization.
    
    """
    def optimize(self, fmax=1.0e-2, steps=1000):
        """
        Optimize a molecular geometry using the LBFGS optimizer
        
        Args:
            fmax (float): Maximum residual force change (default 1.0e-2)
            steps (int): Maximum number of steps (default 1000)
        """
        name = "optimization"
        optimize_file = "%s" % self._get_optimize_path(name)

        optimizer = LBFGS(
                          self.molecule,
                          trajectory="%s.traj" % optimize_file,
                          restart="%s.pkl" % optimize_file)
        optimizer.run(fmax=fmax, steps=steps)
        # Save final geometry in xyz format
        self.save_molecule(name)

    def _get_optimize_path(self, name):
        """Helper method to get optimization path in working directory"""
        return os.path.join(self.working_dir, name)