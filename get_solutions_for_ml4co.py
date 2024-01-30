from pathlib import Path

from pyscipopt import Model
from tqdm import tqdm


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
        model = Model()
        model.hideOutput(True)
        model.readProblem(str(instance_file))

        # Set the time limit and ensure at most 500 solutions are collected
        model.setRealParam('limits/time', 60.0)
        model.setIntParam('limits/maxsol', 500)
        model.setBoolParam('constraints/countsols/collect', True)

        # Generate a new solution for the instance
        model.optimize()

        # Iterate over all solutions found
        for i, sol in enumerate(model.getSols()):
            sol_filename = instance_file.name.replace('.mps.gz', f'_{i}.sol')
            model.writeSol(sol, str(output_dir/sol_filename))
