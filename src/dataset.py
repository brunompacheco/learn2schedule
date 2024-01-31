from pathlib import Path
import numpy as np

from torch.utils.data import Dataset

from src.onts import ONTS


class ONTSFeatures(Dataset):
    def __init__(self, fpaths: list, opt_dir) -> None:
        super().__init__()
        
        opt_dir = Path(opt_dir)

        n_jobs = None
        self.features = list()
        self.targets = list()
        self.fpaths = list()
        for fpath in fpaths:
            instance = ONTS.from_file(fpath)

            if n_jobs is None:
                n_jobs = instance.jobs
            
            assert instance.jobs == n_jobs, ("only a fixed number of jobs "
                                             "(across instances) is supported "
                                             "for now")

            jobs_features = [
                [
                    instance.power_use[j],
                    instance.power_resource[j],
                    instance.min_cpu_time[j],
                    instance.max_cpu_time[j],
                    instance.min_job_period[j],
                    instance.max_job_period[j],
                    instance.min_startup[j],
                    instance.max_startup[j],
                    instance.priority[j],
                    instance.win_min[j],
                    instance.win_max[j],
                ] for j in range(n_jobs)
            ]

            opt_fpath = opt_dir / fpath.name.replace('.json', '_opt.npz')
            # TODO: get only X
            target = np.load(opt_fpath)['arr_3'].astype('uint8')

            self.features.append(np.vstack(jobs_features))
            self.targets.append(target)
            self.fpaths.append(fpath)

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, i):
        # shuffle order of jobs
        return self.features[i], self.targets[i]
