from argparse import Namespace, ArgumentParser
from datetime import datetime
import json
import logging
from pathlib import Path
import wandb
from lightning.pytorch.loggers import WandbLogger

import yaml
import torch
import torchvision
import pytorch_lightning as pl
import pandas as pd
from classification.dataloader import SEGDataModule
from classification import utils_global
from classification.s2_utils import Partitioning, Hierarchy
from classification.model import MultiPartitioningClassifier
from classification.dataset import MsgPackIterableDatasetMultiTargetWithDynLabels 


def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path, default=Path("model_config.yml"))
    return args.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(1)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_params = config["model_params"]
    trainer_params = config["trainer_params"]

    utils_global.check_is_valid_torchvision_architecture(model_params["arch"])

    out_dir = Path(config["out_dir"]) / datetime.now().strftime("%y%m%d-%H%M")
    out_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Output directory: {out_dir}")

    # init classifier
    model = MultiPartitioningClassifier(hparams=Namespace(**model_params))

    # logger = pl.loggers.TensorBoardLogger(save_dir=str(out_dir), name="tb_logs")
    wandb_logger = WandbLogger(log_model="all", save_dir=str(out_dir), name="wb_logs_pretrained", project="Segmentation-Geo-localization")


    checkpoint_dir = out_dir / "ckpts" 
    checkpoint_name = '{epoch}-{val_loss:.4f}'
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint( dirpath=checkpoint_dir, filename=checkpoint_name, every_n_train_steps=1000 )


    trainer = pl.Trainer(
        **trainer_params,
        logger=wandb_logger,
        val_check_interval=model_params["val_check_interval"],
        callbacks=[pl.callbacks.progress.RichProgressBar(), checkpoint_callback],
    )
    
    segData = SEGDataModule( key_img_id= "id", 
                 key_img_encoded=  "image",
                 key_seg_encoded = "seg_encoded",
                 train_label_mapping= "resources/mp16_places365_mapping_h3.json",
                 val_label_mapping = "resources/yfcc_25600_places365_mapping_h3.json",
                 seg_path="/rds/general/user/ao921/ephemeral/Transformer_Based_Geo-localization/resources/r13_local_data/mp16/mp16_seg_images_PNG/",
                 batch_size = 128)
    # trainer.fit(model)
#     trainer.fit(model, datamodule=segData)
    trainer.fit(model, datamodule=segData, ckpt_path="~/GeoEstimation/models/base_M/epoch=014-val_loss=18.4833.ckpt")


if __name__ == "__main__":
    main()
