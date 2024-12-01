import tensorflow as tf
import numpy as np
from pathlib import Path


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min'):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.mode = mode

    def on_epoch_end(self, epoch: int, logs: dict = None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if ((self.mode == 'min' and current < self.best) or
                (self.mode == 'max' and current > self.best)):
            self.best = current
            self.model.save(self.filepath)


class EarlyStoppingWithRestore(tf.keras.callbacks.EarlyStopping):
    def __init__(self,
                 monitor: str = 'val_loss',
                 min_delta: float = 0,
                 patience: int = 0,
                 verbose: int = 0,
                 mode: str = 'auto',
                 baseline: Optional[float] = None,
                 restore_best_weights: bool = True):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights
        )

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.restore_best_weights and self.best is None:
            self.best_weights = self.model.get_weights()

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)