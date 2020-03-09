from typing import List, Dict, Tuple
from collections import namedtuple

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


class Progress:
    """
    Progress class.
    """
    def __init__(self, name, states):
        self.name = name
        self.states = states

        self._ranges = []

    def add_range(self, heuristic_id:HEURISTICS, low, high):
        """
        Takes a heuristic and compares it to range.
        """
        self._ranges.append(
            (heuristic_id, low, high)
        )

    def check_progress(
            self,
            heurisitics:PoseHeuristics
        ) -> Tuple[str, float]:
        progress = []
        for h_id, low, high in self._ranges:
            h_val = heurisitics.get_angle(h_id)
            if h_val is None:
                continue
            
            h_progress = ((h_val - low) * 100) / (high - low)
            progress.append(
                (h_id, h_progress)
            )
        return progress


class Exercise:

    class STATES:
        pass
    
    State = namedtuple('State', 'func progress')

    def __init__(self, name):
        self._states: Dict[str, self.State] = {}
        self._critiques: List[Critique] = []
        self._progresses: List[Progress] = []
        self._init_state = None
        self._rep_transition: tuple = None

        self.name = name
        self.state = None
        self.reps = 0

    def _add_state(self, state, func, progress=None, initial=False):
        if initial:
            self._init_state = state
        self._states[state] = self.State(func, progress)

    def _add_critique(self, critique:Critique):
        self._critiques.append(critique)

    def _add_progress(self, progress:Progress):
        self._progresses.append(progress)

    def _set_rep_transition(self, state1, state2):
        self._rep_transition = (state1, state2)

    def _check_reps(self, curr_state, next_state):
        if self._rep_transition is None:
            raise Exception("Repetition transition not set.")
        return self._rep_transition[0] == curr_state and \
               self._rep_transition[1] == next_state

    def _close_to(self, test_val, target_val, thresh):
        """
        Returns True if `test_val` is within the `thresh` of `target_val`.
        """
        return abs(target_val - test_val) < thresh

    def update(
            self,
            pose: Pose,
            heuristics: PoseHeuristics
        ) -> Tuple[str, List[Critique], List[Progress]]:
        if self.state is None:
            self.state = self._init_state

        critiques = []
        for critique in self._critiques:
            if self.state in critique.states:
                if critique.func(pose, heuristics):
                    critiques.append(critique)

        progress = []
        for p in self._progresses:
            if self.state in p.states:
                for p_i in p.check_progress(heuristics):
                    progress.append(p_i)

        next_state = self._states[self.state].func(pose, heuristics)
        if next_state != self.state:
            if self._check_reps(self.state, next_state):
                self.reps += 1
            self.state = next_state

        return self.state, critiques, progress


from .shoulder_press import ShoulderPress
EXERCISES: Dict[str, Exercise] = {
    'shoulder_press': ShoulderPress
}
