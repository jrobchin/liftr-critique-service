class PoseMetrics():
    """
    This class encapsulates metrics obtained from poses.
    """

    def __init__(self):
        pass

    @classmethod
    def from_human(cls, human):
        metrics = cls()
        metrics.analyze_pose(human)

    def analyze_pose(self, human):
        pass