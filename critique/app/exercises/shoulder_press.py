from critique.pose.modules.pose import Pose, KEYPOINTS
from critique.measure import PoseHeuristics, HEURISTICS, MV_DIRECTIONS
from critique.app.exercises import Exercise, ExerciseState, Critique, Progress


class ShoulderPress(Exercise):

    class STATES:
        SET_UP = 'SET_UP'
        RAISE = 'UP'
        LOWER = 'DOWN'

    def __init__(self):
        super().__init__('Shoulder Press')

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
                self._state_up
            )
        )
        self._add_state(
            ExerciseState(
                self.STATES.LOWER,
                "Lower",
                self._state_down
            )
        )

        self._set_rep_transition(self.STATES.LOWER, self.STATES.RAISE)

        self._add_critique(
            Critique(
                'lock_elbows',
                [self.STATES.RAISE],
                'Make sure not to lock your elbows at the top of your press.',
                self._critique_lock_elbows
            )
        )
        self._add_critique(
            Critique(
                'too_low',
                [self.STATES.LOWER, self.STATES.RAISE],
                'Your arms should make about a 90 degree angle with your body at the bottom.',
                self._critique_too_low
            )
        )

        raise_shoulder_progress = Progress(
            'raise_shoulder',
            [self.STATES.RAISE]
        )
        raise_shoulder_progress.add_range(HEURISTICS.RIGHT_SHLDR, KEYPOINTS.R_SHO, 180, 130)
        raise_shoulder_progress.add_range(HEURISTICS.LEFT_SHLDR, KEYPOINTS.L_SHO, 180, 130)
        self._add_progress(raise_shoulder_progress)

        lower_shoulder_progress = Progress(
            'lower_shoulder',
            [self.STATES.LOWER]
        )
        lower_shoulder_progress.add_range(HEURISTICS.RIGHT_SHLDR, KEYPOINTS.R_SHO, 130, 180)
        lower_shoulder_progress.add_range(HEURISTICS.LEFT_SHLDR, KEYPOINTS.L_SHO, 130, 180)
        self._add_progress(lower_shoulder_progress)

        raise_elbow_progress = Progress(
            'raise_elbow',
            [self.STATES.RAISE]
        )
        raise_elbow_progress.add_range(HEURISTICS.RIGHT_ELBOW, KEYPOINTS.R_ELB, 90, 155)
        raise_elbow_progress.add_range(HEURISTICS.LEFT_ELBOW, KEYPOINTS.L_ELB, 90, 155)
        self._add_progress(raise_elbow_progress)

        lower_elbow_progress = Progress(
            'lower_elbow',
            [self.STATES.LOWER]
        )
        lower_elbow_progress.add_range(HEURISTICS.RIGHT_ELBOW, KEYPOINTS.R_ELB, 155, 90)
        lower_elbow_progress.add_range(HEURISTICS.LEFT_ELBOW, KEYPOINTS.L_ELB, 155, 90)
        self._add_progress(lower_elbow_progress)
        

    # States
    def _state_set_up(self, pose:Pose, heuristics:PoseHeuristics):
        right_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        left_shldr = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        right_elbow = heuristics.get_angle(HEURISTICS.RIGHT_ELBOW)
        left_elbow = heuristics.get_angle(HEURISTICS.LEFT_ELBOW)

        r_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)
        l_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)

        if left_shldr and right_shldr:
            if self._in_range(left_shldr, 180, 20) and \
               self._in_range(right_shldr, 180, 20) and \
               self._in_range(right_elbow, 90, 20) and \
               self._in_range(left_elbow, 90, 20) and \
               r_wri_movement.x == MV_DIRECTIONS.HOLD and \
               r_wri_movement.y == MV_DIRECTIONS.HOLD and \
               l_wri_movement.x == MV_DIRECTIONS.HOLD and \
               l_wri_movement.y == MV_DIRECTIONS.HOLD:
                return self.STATES.RAISE
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
                    return self.STATES.LOWER
        return self.STATES.RAISE

    def _state_down(self, pose:Pose, heuristics:PoseHeuristics):
        left_shldr = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        right_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        if left_shldr and right_shldr:
            if right_shldr > 170 and left_shldr > 170:
                r_wri_movement = heuristics.get_movement(KEYPOINTS.R_WRI)
                l_wri_movement = heuristics.get_movement(KEYPOINTS.L_WRI)
                if r_wri_movement.y == MV_DIRECTIONS.HOLD and \
                   l_wri_movement.y == MV_DIRECTIONS.HOLD:
                    return self.STATES.RAISE
        return self.STATES.LOWER

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
