from critique.pose.modules.pose import Pose, KEYPOINTS
from critique.measure import PoseHeuristics, HEURISTICS, MV_DIRECTIONS
from critique.app.exercises import Exercise, ExerciseState, Critique, Progress


class BicepCurl(Exercise):

    class STATES:
        SET_UP = 'SET_UP'
        RAISE = 'RAISE'
        LOWER = 'LOWER'

    def __init__(self, side='left'):
        super().__init__('Bicep Curl')

        if side in ['left', 'right']:
            self.side = side
        else:
            raise ValueError("`side` must be 'left' or 'right'")

        self._add_state(
            ExerciseState(
                self.STATES.SET_UP,
                "Set up",
                self._state_set_up
            ),
            initial=True
        )
        self._add_state(
            ExerciseState(
                self.STATES.RAISE,
                "Raise",
                self._state_raise
            )
        )
        self._add_state(
            ExerciseState(
                self.STATES.LOWER,
                "Lower",
                self._state_lower
            )
        )

        self._set_rep_transition(self.STATES.RAISE, self.STATES.LOWER)

        self._add_critique(
            Critique(
                'elbow_deviation',
                [self.STATES.RAISE, self.STATES.LOWER],
                'Ensure that you keep your elbow stationary and inline with your shoulder.',
                self._critique_elbow_deviation
            )
        )
        # self._add_critique(
        #     Critique(
        #         'too_low',
        #         [self.STATES.LOWER],
        #         'Your arms should make about a 90 degree angle with your body at the bottom.',
        #         self._critique_too_low
        #     )
        # )

        raise_elbow_progress = Progress(
            'raise_elbow',
            [self.STATES.RAISE]
        )
        raise_elbow_progress.add_range(HEURISTICS.LEFT_ELBOW, KEYPOINTS.L_ELB, 220, 295)
        self._add_progress(raise_elbow_progress)

        lower_elbow_progress = Progress(
            'lower_elbow',
            [self.STATES.LOWER]
        )
        lower_elbow_progress.add_range(HEURISTICS.LEFT_ELBOW, KEYPOINTS.L_ELB, 295, 220)
        self._add_progress(lower_elbow_progress)

    # States
    def _state_set_up(self, pose:Pose, heuristics:PoseHeuristics):
        if self.side == 'left':
            h_shldr = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
            h_elbow = heuristics.get_angle(HEURISTICS.LEFT_ELBOW)
            h_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)
        elif self.side == 'right':
            h_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
            h_elbow = heuristics.get_angle(HEURISTICS.RIGHT_ELBOW)
            h_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)

        if h_shldr and h_elbow and h_wri_movement:
            if self._in_range(h_elbow, 270, 20) and \
               h_wri_movement.x == MV_DIRECTIONS.HOLD and \
               h_wri_movement.y == MV_DIRECTIONS.HOLD:
                return self.STATES.LOWER
        return self.STATES.SET_UP

    def _state_raise(self, pose:Pose, heuristics:PoseHeuristics):
        if self.side == 'left':
            h_elbow = heuristics.get_angle(HEURISTICS.LEFT_ELBOW)
            h_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)
        elif self.side == 'right':
            h_elbow = heuristics.get_angle(HEURISTICS.RIGHT_ELBOW)
            h_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)

        if h_elbow:
            if h_elbow > 295:
                if h_wri_movement.x == MV_DIRECTIONS.HOLD and \
                   h_wri_movement.y == MV_DIRECTIONS.HOLD:
                    return self.STATES.LOWER
        return self.STATES.RAISE

    def _state_lower(self, pose:Pose, heuristics:PoseHeuristics):
        if self.side == 'left':
            h_elbow = heuristics.get_angle(HEURISTICS.LEFT_ELBOW)
            h_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)
        elif self.side == 'right':
            h_elbow = heuristics.get_angle(HEURISTICS.RIGHT_ELBOW)
            h_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)

        if h_elbow:
            if h_elbow < 220:
                h_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)
                if h_wri_movement.x == MV_DIRECTIONS.HOLD and \
                   h_wri_movement.y == MV_DIRECTIONS.HOLD:
                    return self.STATES.RAISE
        return self.STATES.LOWER

    # Critiques
    def _critique_elbow_deviation(self, pose:Pose, heuristics:PoseHeuristics):
        if self.side == 'left':
            elb_pos = pose.keypoints[KEYPOINTS.L_ELB]
            shldr_pos = pose.keypoints[KEYPOINTS.L_SHO]
        elif self.side == 'right':
            elb_pos = pose.keypoints[KEYPOINTS.R_ELB]
            shldr_pos = pose.keypoints[KEYPOINTS.R_SHO]

        if elb_pos[0] == -1:
            return False
        
        return not self._in_range(abs(elb_pos[0]-shldr_pos[0]), 0, 60)

    # def _critique_too_low(self, pose:Pose, heuristics:PoseHeuristics):
    #     r_shldr_angle = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
    #     l_shldr_angle = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
    #     if r_shldr_angle is not None:
    #         return r_shldr_angle > 220
    #     if l_shldr_angle is not None:
    #         return l_shldr_angle > 220
