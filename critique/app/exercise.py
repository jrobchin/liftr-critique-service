from critique.pose.modules.pose import Pose, KEYPOINTS
from critique.measure import PoseHeuristics, HEURISTICS, MV_DIRECTIONS

class Exercise():

    class STATES:
        pass

    def __init__(self, name, initial_state):
        self._states = {}
        self._critiques = []
        self._init_state = initial_state

        self.name = name
        self.state = None
        self.reps = 0
    
    def _add_state(self, state, func):
        self._states[state] = func
    
    def _add_critique(self, states, name, caption, func):
        self._critiques.append((states, name, caption, func))

    def update(self, pose, heuristics):
        raise NotImplementedError()


class ShoulderPress(Exercise):
    
    class STATES:
        UP = 'UP'
        DOWN = 'DOWN'

    def __init__(self):
        super().__init__('Shoulder Press', self.STATES.UP)

        self._add_state(self.STATES.UP, self._state_up)
        self._add_state(self.STATES.DOWN, self._state_down)

        self._add_critique(
            [self.STATES.UP],
            'lock_elbows',
            'Make sure not to lock your elbows at the top of your press.',
            self._critique_lock_elbows
        )
        self._add_critique(
            [self.STATES.DOWN],
            'too_low',
            'Your arms should make about a 90 degree angle with your body at the bottom.',
            self._critique_too_low
        )

    def _state_up(self, pose:Pose, heuristics:PoseHeuristics):
        avg_shldr = heuristics.get_angle(HEURISTICS.AVG_SHLDRS)
        if avg_shldr is not None:
            if avg_shldr < 160:
                r_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)
                l_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)
                if r_wri_movement.y == MV_DIRECTIONS.HOLD and l_wri_movement.y == MV_DIRECTIONS.HOLD:
                    return self.STATES.DOWN
        return self.STATES.UP
    
    def _state_down(self, pose:Pose, heuristics:PoseHeuristics):
        avg_shldr = heuristics.get_angle(HEURISTICS.AVG_SHLDRS)
        if avg_shldr is not None:
            if avg_shldr > 160:
                r_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)
                l_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)
                if r_wri_movement.y == MV_DIRECTIONS.HOLD and l_wri_movement.y == MV_DIRECTIONS.HOLD:
                    return self.STATES.UP
        return self.STATES.DOWN

    def _critique_lock_elbows(self, pose:Pose, heuristics:PoseHeuristics):
        r_elb_angle = heuristics.get_angle(HEURISTICS.RIGHT_ELBOW)
        l_elb_angle = heuristics.get_angle(HEURISTICS.LEFT_ELBOW)
        if r_elb_angle is not None:
            return r_elb_angle > 165
        if r_elb_angle is not None:
            return l_elb_angle > 165

    def _critique_too_low(self, pose:Pose, heuristics:PoseHeuristics):
        pass

    def update(self, pose:Pose, heuristics:PoseHeuristics):
        if self.state == None:
            self.state = self._init_state
        
        critiques = []
        for critique in self._critiques:
            if self.state in critique[0]:
                if critique[3](pose, heuristics):
                    critiques.append((critique[1], critique[2]))

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


EXERCISES = {
    'shoulder_press': ShoulderPress
}