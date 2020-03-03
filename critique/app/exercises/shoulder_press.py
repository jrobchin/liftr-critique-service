from critique.pose.modules.pose import Pose, KEYPOINTS
from critique.measure import PoseHeuristics, HEURISTICS, MV_DIRECTIONS
from critique.app.exercises import Exercise, Critique


class ShoulderPress(Exercise):
    
    class STATES:
        UP = 'UP'
        DOWN = 'DOWN'

    def __init__(self):
        super().__init__('Shoulder Press', self.STATES.UP)

        self._add_state(self.STATES.UP, self._state_up)
        self._add_state(self.STATES.DOWN, self._state_down)

        self._add_critique(
            Critique(
                'lock_elbows',
                [self.STATES.UP],
                'Make sure not to lock your elbows at the top of your press.',
                self._critique_lock_elbows
            )
        )
        self._add_critique(
            Critique(
                'too_low',
                [self.STATES.DOWN, self.STATES.UP],
                'Your arms should make about a 90 degree angle with your body at the bottom.',
                self._critique_too_low
            )
        )

    # States
    def _state_up(self, pose:Pose, heuristics:PoseHeuristics):
        left_shldr = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        right_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        if left_shldr and right_shldr:
            if right_shldr < -50 and left_shldr > 50:
                r_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)
                l_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)
                if r_wri_movement.y == MV_DIRECTIONS.HOLD and l_wri_movement.y == MV_DIRECTIONS.HOLD:
                    return self.STATES.DOWN
        return self.STATES.UP
    
    def _state_down(self, pose:Pose, heuristics:PoseHeuristics):
        left_shldr = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        right_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        if left_shldr and right_shldr:
            if right_shldr > 0 and left_shldr < 0:
                r_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)
                l_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)
                if r_wri_movement.y == MV_DIRECTIONS.HOLD and l_wri_movement.y == MV_DIRECTIONS.HOLD:
                    return self.STATES.UP
        return self.STATES.DOWN

    # Critiques
    def _critique_lock_elbows(self, pose:Pose, heuristics:PoseHeuristics):
        r_elb_angle = heuristics.get_angle(HEURISTICS.RIGHT_ELBOW)
        l_elb_angle = heuristics.get_angle(HEURISTICS.LEFT_ELBOW)
        if r_elb_angle is not None:
            return r_elb_angle > -5
        if r_elb_angle is not None:
            return l_elb_angle < 5

    def _critique_too_low(self, pose:Pose, heuristics:PoseHeuristics):
        r_shldr_angle = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        l_shldr_angle = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        if r_shldr_angle is not None:
            return r_shldr_angle > 65
        if l_shldr_angle is not None:
            return l_shldr_angle < -65