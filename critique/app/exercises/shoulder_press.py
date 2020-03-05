from critique.pose.modules.pose import Pose, KEYPOINTS
from critique.measure import PoseHeuristics, HEURISTICS, MV_DIRECTIONS
from critique.app.exercises import Exercise, Critique


class ShoulderPress(Exercise):

    class STATES:
        SET_UP = 'SET_UP'
        UP = 'UP'
        DOWN = 'DOWN'

    def __init__(self):
        super().__init__('Shoulder Press')

        self._add_state(
            self.STATES.SET_UP,
            self._state_set_up,
            initial=True
        )
        self._add_state(
            self.STATES.UP,
            self._state_up
        )
        self._add_state(
            self.STATES.DOWN,
            self._state_down
        )

        self._set_rep_transition(self.STATES.DOWN, self.STATES.UP)

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
                [self.STATES.DOWN],
                'Your arms should make about a 90 degree angle with your body at the bottom.',
                self._critique_too_low
            )
        )

    # States
    def _state_set_up(self, pose:Pose, heuristics:PoseHeuristics):
        right_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        left_shldr = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        right_elbow = heuristics.get_angle(HEURISTICS.RIGHT_ELBOW)
        left_elbow = heuristics.get_angle(HEURISTICS.LEFT_ELBOW)

        r_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)
        l_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)

        if left_shldr and right_shldr:
            if self._close_to(left_shldr, 180, 20) and \
               self._close_to(right_shldr, 180, 20) and \
               self._close_to(right_elbow, 90, 20) and \
               self._close_to(left_elbow, 90, 20) and \
               r_wri_movement.x == MV_DIRECTIONS.HOLD and \
               r_wri_movement.y == MV_DIRECTIONS.HOLD and \
               l_wri_movement.x == MV_DIRECTIONS.HOLD and \
               l_wri_movement.y == MV_DIRECTIONS.HOLD:
                return self.STATES.UP
        return self.STATES.SET_UP

    def _state_up(self, pose:Pose, heuristics:PoseHeuristics):
        left_shldr = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        right_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        if left_shldr and right_shldr:
            if right_shldr < 130 and left_shldr < 130:
                r_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)
                l_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)
                if r_wri_movement.y == MV_DIRECTIONS.HOLD and \
                   l_wri_movement.y == MV_DIRECTIONS.HOLD:
                    return self.STATES.DOWN
        return self.STATES.UP

    def _state_down(self, pose:Pose, heuristics:PoseHeuristics):
        left_shldr = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        right_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        if left_shldr and right_shldr:
            if right_shldr > 170 and left_shldr > 170:
                r_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)
                l_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)
                if r_wri_movement.y == MV_DIRECTIONS.HOLD and \
                   l_wri_movement.y == MV_DIRECTIONS.HOLD:
                    return self.STATES.UP
        return self.STATES.DOWN

    # Critiques
    def _critique_lock_elbows(self, pose:Pose, heuristics:PoseHeuristics):
        r_elb_angle = heuristics.get_angle(HEURISTICS.RIGHT_ELBOW)
        l_elb_angle = heuristics.get_angle(HEURISTICS.LEFT_ELBOW)
        if r_elb_angle is not None:
            return r_elb_angle > 170
        if r_elb_angle is not None:
            return l_elb_angle > 170

    def _critique_too_low(self, pose:Pose, heuristics:PoseHeuristics):
        r_shldr_angle = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        l_shldr_angle = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        if r_shldr_angle is not None:
            return r_shldr_angle > 220
        if l_shldr_angle is not None:
            return l_shldr_angle > 220
