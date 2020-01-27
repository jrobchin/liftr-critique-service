from critique.measure import HEURISTICS


class State:
    def __init__(self, name, criteria, start=False):
        self.criteria = criteria
        self.start = start


class Exercise():
    def __init__(self, name):
        self.states = []
    
    def _add_state(self, name, criteria):
        start = False
        if len(self.states) == 0:
            start = True
        state = State(name, criteria, start)
        self.states.append(state)

    def update(self, pose, heuristics):
        raise NotImplementedError()


class ShoulderPress(Exercise):
    def __init__(self, name):
        super().__init__(name)

        self._add_state('Up', {
            HEURISTICS.AVG_ELBOWS: None
        })

    def update(self, pose, heuristics):
        pass


class Set():
    def __init__(self, exercise:Exercise):
        self.exercise = exercise
    
    def update(self, pose, heuristics):
        pass