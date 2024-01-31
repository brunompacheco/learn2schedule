import json
import re
from dataclasses import dataclass

import numpy as np
from pyscipopt import Model, quicksum

from .problem import ModelWithPrimalDualCurves


@dataclass
class ONTS:
    """Instance of the ONTS problem."""
    jobs: int
    T: int
    power_use: list  # recurso utilizado por cada tarefa
    power_resource: list  # recurso disponível ao longo da trajetória
    min_cpu_time: list  # mínimo de unidades de tempo que uma tarefa pode consumir em sequência
    max_cpu_time: list  # máximo de unidades de tempo que uma tarefa pode consumir em sequência
    min_job_period: list  # tempo mínimo que uma tarefa deve esperar para se repetir
    max_job_period: list  # tempo máximo que uma tarefa pode esperar para se repetir
    min_startup: list  # tempo mínimo de vezes que uma tarefa pode iniciar
    max_startup: list  # tempo máximo de vezes que uma tarefa pode iniciar
    priority: list  # prioridade de cada tarefa
    win_min: list  # tamanho mínimo da janela
    win_max: list  # tamanho máximo da janela

    subs: int = 1
    initial_soc: float = 0.7
    lower_limit: float = 0.0
    ef: float = 0.9 
    v_bat: float = 3.6 
    q: float = 5
    bat_usage: float = 5

    @classmethod
    def from_file(cls, filepath):
        with open(filepath) as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def random(cls, T, jobs, subs=1):
        orbit_power = np.loadtxt('/home/bruno/sat-gnn/data/raw/resource.csv')

        orbit_start = np.random.randint(0, 600)
        power_resource = orbit_power[orbit_start:orbit_start+T]

        min_power = 0.01
        max_power = 4.00
        power_use = np.random.rand(jobs)
        power_use = (max_power - min_power) * power_use + min_power

        priority = np.arange(jobs) + 1
        np.random.shuffle(priority)

        min_cpu_time = np.random.randint(1, T / 5, size=jobs)
        max_cpu_time = np.random.rand(jobs)
        max_cpu_time = max_cpu_time * (T / 2 - min_cpu_time) + min_cpu_time
        max_cpu_time = max_cpu_time.astype(int)

        max_possible_startup = (T / min_cpu_time).astype(int)
        min_startup = np.random.rand() * (max_possible_startup - 1) + 1
        min_startup = min_startup.astype(int)
        max_startup = np.random.rand(jobs)
        max_startup = max_startup * (T - min_startup) + min_startup
        max_startup = max_startup.astype(int)

        max_possible_job_period = (T / min_startup).astype(int)
        min_job_period = np.random.rand(jobs)
        min_job_period = min_job_period * (max_possible_job_period - 1) + 1
        min_job_period = min_job_period.astype(int)

        max_job_period = np.random.rand(jobs)
        max_job_period = max_job_period * (T - min_job_period) + min_job_period
        max_job_period = max_job_period.astype(int)

        win = np.eye(jobs)[0].astype(int)
        np.random.shuffle(win)
        win_min = win * np.random.randint(1, T * 1/5)
        win_max = win * np.random.randint(T * 4/5, T)
        win_max[win_max == 0] = T

        return cls(
            subs=subs,
            jobs=jobs,
            T=T,
            power_use=power_use.tolist(),
            power_resource=power_resource.tolist(),
            min_cpu_time=min_cpu_time.tolist(),
            max_cpu_time=max_cpu_time.tolist(),
            min_job_period=min_job_period.tolist(),
            max_job_period=max_job_period.tolist(),
            min_startup=min_startup.tolist(),
            max_startup=max_startup.tolist(),
            priority=priority.tolist(),
            win_min=win_min.tolist(),
            win_max=win_max.tolist(),
        )

    @property
    def vars_names(self):
        if hasattr(self, '_vars_names'):
            return self._vars_names
        else:
            model = self.to_gurobipy()

            vars_names = np.core.defchararray.array([v.VarName for v in model.getVars()])
            vars_names = vars_names[(vars_names.find('x(') >= 0) | (vars_names.find('phi(') >= 0)]
            self._vars_names = vars_names

            return self.vars_names

    def add_phi_to_candidate(self, candidate: dict):
        X = np.zeros((self.jobs, self.T), dtype='uint8')
        for var_name, var_x in candidate.items():
            j, t = re.fullmatch(r"x\((\d+),(\d+)\)", var_name).groups()
            j = int(j)
            t = int(t)
            X[j,t] = np.round(var_x)

        Phi = X.copy()
        Phi[:,1:] = np.where(X[:,1:] - X[:,:-1] != 0, X[:,1:], np.zeros_like(X[:,1:]))

        full_candidate = dict()
        for var_name in self.vars_names:
            var, j, t = re.fullmatch(r"(x|phi)\((\d+),(\d+)\)", var_name).groups()
            j = int(j)
            t = int(t)

            if var == 'x':
                full_candidate[var_name] = X[j,t]
            elif var == 'phi':
                full_candidate[var_name] = Phi[j,t]

        return full_candidate

    def to_scip(self, coupling=True, new_inequalities=False, timeout=60,
                enable_primal_dual_integral=True) -> Model:
        # create a model
        if enable_primal_dual_integral:
            model = ModelWithPrimalDualCurves()
        else:
            model = Model()

        if timeout is not None:
            model.setParam('limits/time', timeout)

        # create decision variables
        x = {}
        phi = {}
        for j in range(self.jobs):
            # the order in which we add the variables matter. I want all
            # variables associated with a given job to be together, like
            # x(0,0),...,x(0,T-1),phi(0,0),...phi(0,T-1),x(1,0),...,x(1,T-1),phi(1,0),...
            for t in range(self.T):
                    x[j,t] = model.addVar(name="x(%s,%s)" % (j, t), lb=0, ub=1,
                                          vtype="BINARY")
            for t in range(self.T):
                    phi[j,t] = model.addVar(name="phi(%s,%s)" % (j, t), lb=0, ub=1,
                                            vtype="BINARY")

        model.setObjective(quicksum(self.priority[j] * x[j,t]
                                    for j in range(self.jobs)
                                    for t in range(self.T)),
                           "maximize")

        # phi defines startups of jobs
        for t in range(self.T):
            for j in range(self.jobs):
                if t == 0:
                        model.addCons(phi[j,t] >= x[j,t] - 0, f"C1_j{j}_t{t}")
                else:
                        model.addCons(phi[j,t] >= x[j,t] - x[j,t - 1], f"C2_j{j}_t{t}")

                model.addCons(phi[j,t] <= x[j,t], f"C3_j{j}_t{t}")

                if t == 0:
                        model.addCons(phi[j,t] <= 2 - x[j,t] - 0, f"C4_j{j}_t{t}")
                else:
                        model.addCons(phi[j,t] <= 2 - x[j,t] - x[j,t - 1], f"C5_j{j}_t{t}")

        # minimum and maximum number of startups of a job
        for j in range(self.jobs):
            model.addCons(quicksum(phi[j,t] for t in range(self.T)) >= self.min_startup[j], f"C6_j{j}")
            model.addCons(quicksum(phi[j,t] for t in range(self.T)) <= self.max_startup[j], f"C7_j{j}")

            ###############################
            # precisa ajustar

            # execution window
            model.addCons(quicksum(x[j,t] for t in range(self.win_min[j])) == 0, f"C8_j{j}")
            model.addCons(quicksum(x[j,t] for t in range(self.win_max[j], self.T)) == 0, f"C9_j{j}")

        for j in range(self.jobs):
            # minimum period between jobs
            for t in range(0, self.T - self.min_job_period[j] + 1):
                model.addCons(quicksum(phi[j,t_] for t_ in range(t, t + self.min_job_period[j])) <= 1, f"C10_j{j}_t{t}")

            # periodo máximo entre jobs
            for t in range(0, self.T - self.max_job_period[j] + 1):
                model.addCons(quicksum(phi[j,t_] for t_ in range(t, t + self.max_job_period[j])) >= 1, f"C11_j{j}_t{t}")

            # min_cpu_time das jobs
            for t in range(0, self.T - self.min_cpu_time[j] + 1):
                model.addCons(quicksum(x[j,t_] for t_ in range(t, t + self.min_cpu_time[j])) >= self.min_cpu_time[j] * phi[j,t], f"C12_j{j}_t{t}")

            # max_cpu_time das jobs
            for t in range(0, self.T - self.max_cpu_time[j]):
                    model.addCons(quicksum(x[j,t_] for t_ in range(t, t + self.max_cpu_time[j] + 1)) <= self.max_cpu_time[j], f"C13_j{j}_t{t}")

            # min_cpu_time no final do periodo
            for t in range(self.T - self.min_cpu_time[j] + 1, self.T):
                    model.addCons(quicksum(x[j,t_] for t_ in range(t, self.T)) >= (self.T - t) * phi[j,t], f"C14_j{j}_t{t}")

        if coupling:
            soc = {}
            i = {}
            b = {}
            for t in range(self.T):
                soc[t] = model.addVar(vtype="CONTINUOUS", name="soc(%s)" % t)
                i[t] = model.addVar(lb=None, vtype="CONTINUOUS", name="i(%s)" % t)
                b[t] = model.addVar(lb=None, vtype="CONTINUOUS", name="b(%s)" % t)

            ################################
            # Add power constraints
            for t in range(self.T):
                model.addCons(quicksum(self.power_use[j] * x[j,t] for j in range(self.jobs)) <= self.power_resource[t] + self.bat_usage * self.v_bat, f"C15_t{t}")# * (1 - alpha[t]))

            ################################
            # Bateria
            ################################
            for t in range(self.T):
                model.addCons(quicksum(self.power_use[j] * x[j,t] for j in range(self.jobs)) + b[t] == self.power_resource[t], f"C16_t{t}")

            # Define the i_t, SoC_t constraints in Gurobi
            for t in range(self.T):
                # P = V * I 
                model.addCons(b[t] / self.v_bat >= i[t], f"C17_t{t}")

                if t == 0:
                    # SoC(1) = SoC(0) + p_carga[1]/60
                    model.addCons(soc[t] == self.initial_soc + (self.ef / self.q) * (i[t] / 60), f"C18_t{t}")
                else:
                    # SoC(t) = SoC(t-1) + (ef / Q) * I(t)
                    model.addCons(soc[t] == soc[t - 1] + (self.ef / self.q) * (i[t] / 60), f"C19_t{t}")

                # Set the lower and upper limits on SoC
                model.addCons(self.lower_limit <= soc[t], f"C20_t{t}")
                model.addCons(soc[t] <= 1, f"C21_t{t}")

        if new_inequalities:
            # first
            for j in range(self.jobs):
                for t in range(self.T):
                    model.addCons(
                        quicksum(phi[j, t_] for t_ in range(
                            t,
                            min(self.T, t + self.min_cpu_time[j] + 1)
                        )) <= 1,
                        name=f"VI_min_CPU_TIME_phi({j},{t})"
                    )

            # third
            for j in range(self.jobs):
                model.addCons(
                    quicksum(x[j, t] for t in range(self.T)) <=
                        self.max_cpu_time[j] * quicksum(
                            phi[j, t] for t in range(self.T)
                        ),
                    name = f"VI_max_cpu_time_2({j})"
                )

            # fourth
            for j in range(self.jobs):
                for t in range(0, self.T - self.max_cpu_time[j], 1):
                    model.addCons(
                        quicksum(x[j, t_] for t_ in range(
                                t, t + self.max_cpu_time[j], 1
                            )) <= self.max_cpu_time[j] * quicksum(
                                phi[j, t_] for t_ in range(
                                    max(t - self.max_cpu_time[j] + 1,0),
                                    t + self.max_cpu_time[j],
                                    1,
                                )
                            ),
                        name = f"VI_max_cpu_time_3({j},{t})"
                    )

            # fifth
            for j in range(self.jobs):
                for t in range(0, self.T - self.min_job_period[j] + 1):
                    model.addCons(quicksum(x[j, t_] for t_ in range(t, t + self.min_job_period[j])) <= self.min_job_period[j],
                                    name = f"VI_min_period_btw_jobs_2({j},{t})")

            # sixth
            for j in range(self.jobs):
                if self.max_cpu_time[j] < (self.max_job_period[j] - self.min_cpu_time[j]):
                    for t in range(0, self.T - self.max_cpu_time[j]):
                        model.addCons(phi[j,t] + x[j, t + self.max_cpu_time[j]] <= 1,
                                    name=f"VI_max_period_btw_jobs({j},{t})")

        return model

    def to_json(self, fp):
        with open(fp, 'w') as f:
            json.dump(self, f, default=lambda o: {k:v for k,v in o.__dict__.items()
                                                  if k != '_vars_names'})
