from pathlib import Path

import numpy as np
import torch

from src.dataset import ONTSFeatures
from src.net import ONTSMLP
from src.trainer import Trainer
from src.utils import debugger_is_active


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if debugger_is_active():
        import random
        seed = 33
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)

        wandb_project = None  # avoid logging run

        torch.autograd.set_detect_anomaly(True)
    else:
        seed = None
        # wandb_project = 'PROJECT-NAME'
        wandb_project = "milp-solution-prediction"

    J = 9
    opt_dir = Path('data/interim/ONTS')
    instances_dir = Path('data/raw/ONTS')
    train_instances = [fp for fp in instances_dir.glob('*.json')
                       if (int(fp.stem.split('_')[1]) == J)
                       and (int(fp.stem.split('_')[2]) < 160)]
    val_instances = [fp for fp in instances_dir.glob('*.json')
                     if (int(fp.stem.split('_')[1]) == J)
                     and (int(fp.stem.split('_')[2]) >= 160)]
    train_dataset = ONTSFeatures(train_instances, opt_dir)
    val_dataset = ONTSFeatures(val_instances, opt_dir)

    Trainer(
        ONTSMLP(J, 125, 11, 3, 50).double(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        wandb_project=wandb_project,
        wandb_group="TEST",
        epochs=20,
        random_seed=seed,
        device=device,
    ).run()
