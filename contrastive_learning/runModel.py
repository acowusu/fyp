import torch
from prebaked_dataloader import ImageMapDataModule
from model import SimCLR
simCLR = SimCLR.load_from_checkpoint("~/contrastive_learning/model_assets/240530-0337/ckpts/epoch=0-val_loss=0.0000.ckpt")
simCLR.eval()

data_module = ImageMapDataModule(batch_size=1)
data_module.setup("fit")
dataloader = data_module.train_dataloader()

for X, Y in dataloader:
        with torch.no_grad():
            print(X.shape)
            y_hat = simCLR(X)
            print(y_hat)