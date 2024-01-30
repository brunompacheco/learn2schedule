from pyscipopt import SCIP_EVENTTYPE, Model
from pyscipopt.scip import Event, Eventhdlr


class PrimalDualTracker(Eventhdlr):
    def __init__(self, eventtypes=[SCIP_EVENTTYPE.NODESOLVED,
                                   SCIP_EVENTTYPE.BESTSOLFOUND]):
        """Registers primal and dual bounds on `eventtypes` events.
        """
        self.eventtypes = eventtypes

        self._eventtype = 0
        for eventtype in self.eventtypes:
            self._eventtype |= eventtype

        self.primals = list()
        self.duals = list()
        self.times = list()

        self.calls = list()

    def eventinit(self):
        self.model.catchEvent(self._eventtype, self)

    def eventexit(self):
        self.model.dropEvent(self._eventtype, self)

    def eventexec(self, event: Event):
        try:
            self.times.append(self.model.getTotalTime())

            # primal = self.model.getPrimalbound()
            # if primal < self.primals[0]:
            #     primal = self.primals[0]
            # self.primals.append(primal)

            self.primals.append(self.model.getPrimalbound())
            self.duals.append(self.model.getDualbound())
        except Exception:
            pass

class ModelWithPrimalDualCurves(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.primal_dual_tracker = PrimalDualTracker()
        self.includeEventhdlr(self.primal_dual_tracker, "Primal Dual Tracker",
                              "Catches changes in the primal and the dual bounds")

    def get_primal_curve(self):
        return self.primal_dual_tracker.times, self.primal_dual_tracker.primals

    def get_dual_curve(self):
        return self.primal_dual_tracker.times, self.primal_dual_tracker.duals

def load_model(mps_file) -> Model:
    model = ModelWithPrimalDualCurves()

    model.readProblem(str(mps_file))

    return model
