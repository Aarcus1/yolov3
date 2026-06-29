from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Optional, Dict
from source.utilities.subset_type import SplitType

class Logger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None
        self.train_bar: Optional[tqdm] = None
        self.validation_bar: Optional[tqdm] = None
        self.test_bar: Optional[tqdm] = None

    def __enter__(self):
        self.writer = SummaryWriter(self.log_dir)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.writer:
            self.writer.close()
        if self.train_bar:
            self.train_bar.close()
        if self.test_bar:
            self.test_bar.close()
        if self.validation_bar:
            self.validation_bar.close()

    def init_train_bar(self, total_steps: int, starting_steps: int = 0):
        if self.train_bar is None:
            self.train_bar = tqdm(total=total_steps, desc="Training", position=0, initial=starting_steps)
    def init_test_bar(self, total_steps: int, starting_steps: int = 0):
        if self.test_bar is None:
            self.test_bar = tqdm(total=total_steps, desc="Testing", position=0, initial=starting_steps)

    def init_validation_bar(self, total_steps: int):
        if self.validation_bar is None:
            self.validation_bar = tqdm(
                total=total_steps,
                desc="Validation",
                position=1,
                leave=True,
            )
        else:
            self.validation_bar.reset(total=total_steps)

    def close_train_bar(self):
        if self.train_bar is not None:
            self.train_bar.close()
            self.train_bar = None

    def close_validation_bar(self):
        if self.validation_bar is not None:
            self.validation_bar.close()
            self.validation_bar = None

    def _log_iteration(self, metrics: Dict[str, float], iteration_number: int, split_type: SplitType, iteration_type: str):
        for metric_name, value in metrics.items():
            full_name = f"{split_type.name}/{iteration_type}/{metric_name}"
            self.writer.add_scalar(full_name, value, iteration_number)
            self.writer.flush()

    def log_step(self, metrics: Dict[str, float], step: int, split_type: SplitType):
        if split_type == SplitType.TEST:
            self.test_bar.update(1)
            return
        self._log_iteration(metrics, step, split_type, "step")
        if split_type == SplitType.TRAIN:
            self.train_bar.update(1)


    def log_epoch(self, metrics: Dict[str, float], step: int, split_type: SplitType):
        self._log_iteration(metrics, step, split_type, "epoch")
        if split_type == SplitType.VALIDATION:
            self.validation_bar.update(1)
