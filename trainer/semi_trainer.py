from .trainer import Trainer
class SemiTrainer(Trainer):
    def __init__(self, opt, device, LOCAL_RANK, RANK, WORLD_SIZE) -> None:
        super().__init__()