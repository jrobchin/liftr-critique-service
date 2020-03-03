from typing import List, Dict

from critique.pose.modules.pose import Pose, KEYPOINTS
from critique.measure import PoseHeuristics, HEURISTICS, MV_DIRECTIONS


class Critique:
    def __init__(self, name, states, msg, func):
        self.name = name
        self.states = states
        self.msg = msg
        self.func = func

    def __call__(self, pose: Pose, heuristics: PoseHeuristics):
        self.func(pose, heuristics)


class Exercise:

    class STATES:
        pass

    def __init__(self, name):
        self._states = {}
        self._critiques: List[Critique] = []
        self._init_state = None

        self.name = name
        self.state = None
        self.reps = 0

    def _add_state(self, state, func, initial=False):
        if initial:
            self._init_state = state
        self._states[state] = func

    def _add_critique(self, critique: Critique):
        self._critiques.append(critique)

    def _close_to(self, test_val, target_val, thresh):
        """
        Returns True if `test_val` is within the `thresh` of `target_val`.
        """
        return abs(target_val - test_val) < thresh


class Set():
    """
    Set class.
    """
    def __init__(self, exercise:Exercise):
        self._exercise = exercise
        self._state = self._exercise.state

    def update(self, pose, heuristics):
        pass


from .shoulder_press import ShoulderPress
EXERCISES: Dict[str, Exercise] = {
    'shoulder_press': ShoulderPress
}
