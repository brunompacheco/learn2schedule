from pathlib import Path
import numpy as np

from tqdm import tqdm

from src.problem import load_model


if __name__ == '__main__':
    # Define the directory paths
    input_dir = Path('data/raw/instances/3_anonymous')
    output_dir = Path('data/interim/solutions/3_anonymous')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get the list of instance files in the input directory
    instance_files = list(input_dir.glob('**/*.mps.gz'))

    # Iterate over each instance file
    for instance_file in tqdm(instance_files):
        # Load the instance into a SCIP model
        model = load_model(str(instance_file))

        # Set the time limit and ensure at most 500 solutions are collected
        model.setRealParam('limits/time', 5*60.0)
        model.setIntParam('limits/maxsol', 500)
        model.setBoolParam('constraints/countsols/collect', True)

        # Generate a new solution for the instance
        model.optimize()

        # Iterate over all solutions found
        for i, sol in enumerate(model.getSols()):
            sol_filename = instance_file.name.replace('.mps.gz', f'_{i}.sol')
            model.writeSol(sol, str(output_dir/sol_filename))

        # Save the primal/dual bound curve
        curve_filename = instance_file.name.replace('.mps.gz', f'_bounds.npz')
        primal_t, primal_x = model.get_primal_curve()
        dual_t, dual_x = model.get_dual_curve()

        np.savez(output_dir/curve_filename, primal_t=primal_t,
                 primal_x=primal_x, dual_t=dual_t, dual_x=dual_x)
