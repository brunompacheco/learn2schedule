from pathlib import Path

import pandas as pd

from pyscipopt import Model
from tqdm import tqdm


def are_solutions_equal(m, sol1, sol2):
    for var in m.getVars():
        if m.getSolVal(sol1, var) != m.getSolVal(sol2, var):
            return False
    return True

if __name__ == '__main__':
    miplib_dir = Path('./data/raw/MIPLIB')
    sols_dir = Path('./data/interim/MIPLIB/solutions')
    sols_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(miplib_dir / 'The Collection Set.csv')
    df.columns = [col.split(' ')[0] for col in df.columns]
    df = df[df['Tags'].str.contains('set_packing')]

    for instance in tqdm(df.iloc):
        name = instance['Instance']
        instance_fp = next(miplib_dir.glob(f"**/{name}.mps.gz"))

        model = Model()
        model.hideOutput(True)
        
        model.readProblem(str(instance_fp))
        model.setRealParam('limits/time', 2 * 60.0)

        solutions = [
            model.readSolFile(str(sol_fp))
            for sol_fp in miplib_dir.glob(f'**/{name}.sol.gz')
        ]

        if instance['Status'] == 'open':
            # only add solutions if the problem is really hard
            for sol in solutions:
                model.addSol(sol, False)

        model.optimize()

        for i, new_sol in enumerate(model.getSols()):
            skip = False
            for sol in solutions:
                if are_solutions_equal(model, sol, new_sol):
                    skip = True
                    break
            if skip:
                continue

            model.writeSol(new_sol, str(sols_dir / f'{name}_{i}.sol.gz'))
        
        for j, sol in enumerate(solutions):
            model.writeSol(sol, str(sols_dir / f'{name}_{i+j+1}.sol.gz'))
