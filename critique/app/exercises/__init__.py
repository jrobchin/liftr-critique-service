from typing import List, Dict

from critique.pose.modules.pose import Pose, KEYPOINTS
from critique.measure import PoseHeuristics, HEURISTICS, MV_DIRECTIONS


class Critique:
    def __init__(self, name, states, msg, func):
        self.name = name
        self.states = states
        self.msg = msg
        self.func = func

    def __call__(self, pose:Pose, heuristics:PoseHeuristics):
        self.func(pose, heuristics)


class Exercise:

    class STATES:
        pass

    def __init__(self, name, initial_state):
        self._states = {}
        self._critiques: List[Critique] = []
        self._init_state = initial_state

        self.name = name
        self.state = None
        self.reps = 0
    
    def _add_state(self, state, func):
        self._states[state] = func
    
    def _add_critique(self, critique:Critique):
        self._critiques.append(critique)

    def update(self, pose:Pose, heuristics:PoseHeuristics):
        if self.state == None:
            self.state = self._init_state
        
        critiques = []
        for critique in self._critiques:
            if self.state in critique.states:
                if critique.func(pose, heuristics):
                    critiques.append(critique)

        next_state = self._states[self.state](pose, heuristics)
        if next_state != self.state:
            self.state = next_state
            if self.state == self._init_state:
                self.reps += 1
        
        return self.state, critiques


class Set():
    def __init__(self, exercise:Exercise):
        self._exercise = exercise
        self._state = self._exercise.state
    
    def update(self, pose, heuristics):
        pass


from .shoulder_press import ShoulderPress
EXERCISES: Dict[str, Exercise] = {
    'shoulder_press': ShoulderPress
}