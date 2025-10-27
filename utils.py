import warnings
import math
import json
import time
import os
from pathlib import Path
from skimage.metrics import structural_similarity as compare_ssim
from numpy import ndarray
import torch.nn as nn
from torch import inf
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


# monai.losses.ssim_loss can be used as a metric
# but in order to match numbers with the fastMRI leaderboard,
# we use scikit-image ssim metric
def skimage_ssim(gt: ndarray, rec: ndarray) -> float:
    """
    Compute SSIM quality assessment metric
    inputs:
        gt: ground truth with the last two dims being spatial and the number of slices
        rec: reconstruction with the same dimensions as gt
    outputs:
        skimage SSIM score between gt and rec
    """
    # assumes 3D inputs
    return compare_ssim(gt.transpose(1, 2, 0), rec.transpose(1, 2, 0), channel_axis=2, data_range=gt.max())


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:  # 添加 bias 是否存在的检查
                nn.init.constant_(m.bias, 0)


class GradualWarmupScheduler(LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        opOptimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class CosineAnnealingWarmRestartsOnPlateau(LRScheduler):
    def __init__(
        self,
        optimizer,
        T_max,
        min_lr,
        max_lr,
        factor=2,
        mode='min',
        patience=10,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=0,
        last_epoch=-1,
        verbose=False
    ):
        if T_max <= 0 or not isinstance(T_max, int):
            raise ValueError(f"Expected positive integer T_max, but got {T_max}")
        if not isinstance(min_lr, (float, int)):
            raise ValueError(f"Expected float or int eta_min, but got {min_lr} of type {type(min_lr)}")
        self.T = T_max
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.T_cur = last_epoch
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.last_epoch = 0
        self.verbose = verbose
        self.optimizer = optimizer
        self.best = None
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self.base_lr = self._last_lr.copy()
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [self.min_lr + (lr  - self.min_lr) * (1 + math.cos(math.pi * self.T_cur / self.T)) / 2
                for lr in self.base_lr]

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def step(self, metrics, epoch=None):
        current = float(metrics)

        if self.last_epoch < 0:
            epoch = 0
        else:
            epoch = self.last_epoch + 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.base_lr = [min(self.max_lr, lr * self.factor) for lr in self._last_lr]
            if self.verbose:
                print(f"Epoch {epoch}: reducing LR from {self._last_lr} to {self.get_lr()}")
            self.T_cur = -1
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self.T_cur = self.T_cur + 1  # update T_cur in CosAnnelingRestarts

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = min(self.max_lr, lr)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

def save_sourcecode(code_rootdir: Path, out_dir: Path, additional_files=None, file_type='*.py', verbose=True):
    if not os.path.isdir(code_rootdir):
        raise FileNotFoundError(f'Code root dir not exists! {code_rootdir}')
    if verbose:
        print('Backup source code under root_dir:', code_rootdir)

    out_dir.mkdir(parents=True, exist_ok=True)
    outpath = out_dir / f"source_code_{time.strftime('%m%d_%H%M')}.tar"
    tar_opt = 'cvf' if verbose else 'cf'
    # os.system(f"find {code_rootdir} -name '{file_type}' | tar -{tar_opt} {outpath} -T -")
    files_to_add = []

    # Find all matching files in the code_rootdir
    for file in code_rootdir.rglob(file_type):
        files_to_add.append(str(file))

    # Add additional specific files
    if additional_files:
        for file in additional_files:
            if os.path.isfile(file):
                files_to_add.append(str(file))
            else:
                print(f"Warning: Additional file not found - {file}")

    # Write files to tar
    file_list = ' '.join(files_to_add)
    os.system(f"tar -{tar_opt} {outpath} {file_list}")

    if verbose:
        print(f"Source code backed up to {outpath}")