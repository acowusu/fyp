import os
from prebaked_dataloader import ImageMapDataModule
from model import SimCLR
# from pl_bolts.callbacks.self_supervised import SSLOnlineEvaluator
import pytorch_lightning as pl
import wandb
from lightning.pytorch.loggers import WandbLogger
from argparse import Namespace, ArgumentParser
from pathlib import Path
import logging
import yaml
from datetime import datetime
# from lightning.pytorch.callbacks import DeviceStatsMonitor

def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path, default=Path("./config.yml"))
    return args.parse_args()





def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(1)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)



    out_dir = Path(config["out_dir"]) / datetime.now().strftime("%y%m%d-%H%M")
    out_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Output directory: {out_dir}")


    wandb_logger = WandbLogger(log_model="all", save_dir=str(out_dir), name="wb_logs", project="Geometric-Geo-localization")
    checkpoint_dir = out_dir / "ckpts" 
    checkpoint_name = '{epoch}-{val_loss:.4f}'
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint( dirpath=checkpoint_dir, filename=checkpoint_name, every_n_train_steps=100 )


    callbacks = [ checkpoint_callback, pl.callbacks.progress.RichProgressBar(), pl.callbacks.DeviceStatsMonitor()]

    model_params = config["model_params"]
    trainer_params = config["trainer_params"]
    loader_params = config["loader_params"]

    # init data
    dm = ImageMapDataModule(batch_size=model_params["batch_size"],  **loader_params)

    # realize the data
    dm.prepare_data()
    dm.setup("fit")

    model = SimCLR(**model_params )  
    trainer = pl.Trainer( **trainer_params , callbacks=callbacks, logger=wandb_logger)
    trainer.fit(model,  datamodule=dm)


if __name__ == "__main__":
    main()
