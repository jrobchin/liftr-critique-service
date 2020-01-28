from critique.measure import HEURISTICS


class State:
    def __init__(self, name, criteria, start=False):
        self.criteria = criteria
        self.start = start


class Exercise():
    def __init__(self, name):
        self.states = {}
        self.trasitions = {}

    def update(self, pose, heuristics):
        raise NotImplementedError()


class ShoulderPress(Exercise):
    def __init__(self, name):
        super().__init__(name)

    def update(self, pose, heuristics):
        pass


class Set():
    def __init__(self, exercise:Exercise):
        self.exercise = exercise
    
    def update(self, pose, heuristics):
        pass