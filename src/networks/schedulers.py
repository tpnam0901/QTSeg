from torch import optim
from configs.base import Config


def StepLR(optimizer, cfg: Config):
    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.lr_step_size,
        gamma=cfg.lr_step_gamma,
        last_epoch=cfg.scheduler_last_epoch,
    )


def MultiStepLR(optimizer, cfg: Config):
    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.lr_milestones,
        gamma=cfg.lr_multistep_gamma,
        last_epoch=cfg.scheduler_last_epoch,
    )


def ExponentialLR(optimizer, cfg: Config):
    return optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.lr_exp_gamma,
        last_epoch=cfg.scheduler_last_epoch,
    )


def CosineAnnealingLR(optimizer, cfg: Config):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.lr_T_max,
        eta_min=cfg.lr_eta_min,
    )


def ReduceLROnPlateau(optimizer, cfg: Config):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.lr_plateau_mode,
        factor=cfg.lr_plateau_factor,
        patience=cfg.lr_plateau_patience,
        threshold=cfg.lr_plateau_threshold,
        threshold_mode=cfg.lr_plateau_threshold_mode,
        cooldown=cfg.lr_plateau_cooldown,
        min_lr=cfg.lr_plateau_min_lr,
        eps=cfg.lr_plateau_eps,
    )


def CosineAnnealingWarmRestarts(optimizer, cfg: Config):
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.lr_T_0,
        T_mult=cfg.lr_T_mult,
        eta_min=cfg.lr_eta_min,
        last_epoch=cfg.scheduler_last_epoch,
    )


class IdentityScheduler:
    def __init__(self, optimizer, cfg: Config):
        pass

    def step(self):
        pass

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass


class PolyLR:
    def __init__(self, optimizer, cfg: Config):
        self.optimizer = optimizer
        self.max_iter = cfg.epochs
        self.init_lr = cfg.learning_rate
        self.exponent = 0.9
        self.current_iter = 0

    def step(self):
        lr = self.init_lr * (1 - self.current_iter / self.max_iter) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_iter += 1

    def state_dict(self):
        return {"current_iter": self.current_iter}

    def load_state_dict(self, state_dict):
        self.current_iter = state_dict["current_iter"]
