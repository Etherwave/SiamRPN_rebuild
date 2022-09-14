class Eval_Standard:

    def __init__(self):
        self.wait_number = 5
        self.burn_in_period = 10
        self.failure_iou_threshold = 0.1

        self.total_iou = 0
        self.success_track_frames_cnt = 0
        self.average_iou = 0
        self.failure_cnt = 0
        self.robustness = 0
