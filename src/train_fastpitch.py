import lightning.pytorch as pl
import torch
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig

from models.fastspeech import FastSpeechModel

torch.set_float32_matmul_precision('medium')

@hydra_runner(config_path="configs", config_name="")
def main(cfg:DictConfig):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = FastSpeechModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


if __name__ == '__main__':
    main()
