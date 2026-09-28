from typing import List, Any

import hydra
import lightning as L
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Timer
from omegaconf import DictConfig, OmegaConf, ListConfig, base
import torch
from collections import defaultdict
OmegaConf.register_new_resolver("getIndex", lambda lst, idx: lst[idx])

@hydra.main(version_base="1.3", config_path="./configs", config_name="eval")
def main(cfg: DictConfig):
    """
    Entry point for training the model.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    loggers: List[Logger] = [
        hydra.utils.instantiate(logger) for logger in cfg.get("loggers").values()
    ]

    # Log hyperparameters
    for logger in loggers:
        logger.log_hyperparams(OmegaConf.to_object(cfg))

    # Initialize the model
    model: L.LightningModule = hydra.utils.instantiate(cfg.get("model"))
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.get("data"))
    callbacks: List[L.Callback] = [
        hydra.utils.instantiate(callback) for callback in cfg.get("callbacks").values()
    ]

    timer = Timer()
    callbacks.append(timer)

    # Initialize the trainer
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.get("trainer"), callbacks=callbacks, logger=loggers
    )

    ckpt_path = cfg.get("ckpt_path")
    if cfg.get("lora_ckpt_path"):
        # A fine-tune checkpoint holds only the adapter weights (on_save_checkpoint),
        # so load base GARF first and put the adapter on top, as app.py does.
        model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        )
        model.enable_lora(cfg.get("lora_ckpt_path"))
        ckpt_path = None

    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    print("Time taken: ", timer.time_elapsed("test"))


if __name__ == "__main__":
    main()
