import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np
from typing import Optional
import pytorch_lightning as pl
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from transformers import AutoImageProcessor
from transformers import AutoConfig, AutoModel
import math


# def nt_xent_loss(out_1, out_2, temperature):
#     out = torch.cat([out_1, out_2], dim=0)
#     n_samples = len(out)

#     # Full similarity matrix
#     cov = torch.mm(out, out.t().contiguous())
#     sim = torch.exp(cov / temperature)

#     mask = ~torch.eye(n_samples, device=sim.device).bool()
#     neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

#     # Positive similarity
#     pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
#     pos = torch.cat([pos, pos], dim=0)

#     loss = -torch.log(pos / neg).mean()
#     return loss

class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input.contiguous(), op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


def nt_xent_loss(out_1, out_2, temperature, eps=1e-6):
    """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
    """
    # gather representations in case of distributed training
    # out_1_dist: [batch_size * world_size, dim]
    # out_2_dist: [batch_size * world_size, dim]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        out_1_dist = SyncFunction.apply(out_1)
        out_2_dist = SyncFunction.apply(out_2)
    else:
        out_1_dist = out_1
        out_2_dist = out_2

    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss


class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)
    

class SimCLR(pl.LightningModule):
    def __init__(self,
                 batch_size,
                 warmup_epochs=10,
                 lr=1e-4,
                 opt_weight_decay=1e-6,
                 loss_temperature=0.5,
                 model_name: str = "google/efficientnet-b0",

                 **kwargs):
        """
        Args:
            batch_size: the batch size
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.nt_xent_loss = nt_xent_loss
        self.encoder = self.init_encoder()

        # h -> || -> z
        input_dim = self.encoder.config.hidden_dim # 2048 for resnets and 1280 for efficientnets
        self.projection = Projection(input_dim=input_dim)

    def init_encoder(self):
        # config = AutoConfig.from_pretrained(self.hparams.model_name)
        # encoder = AutoModelForImageClassification.from_config(config)
        encoder = AutoModel.from_pretrained(self.hparams.model_name)
        # encoder = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0")
        print("Encoder loaded")
        # encoder = resnet50_bn(return_all_feature_maps=False)

        # # when using cifar10, replace the first conv so image doesn't shrink away
        # encoder.conv1 = nn.Conv2d(
        #     3, 64,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=False
        # )
        return encoder

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]

    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.hparams.batch_size

    def configure_optimizers(self):
        print(self.hparams.optim["params"])
        optim_feature_extrator = torch.optim.SGD(
            self.parameters(), **self.hparams.optim["params"]
        )

        return {
            "optimizer": optim_feature_extrator,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optim_feature_extrator, **self.hparams.scheduler["params"]
                ),
                "interval": "epoch",
                "name": "lr",
            },
        }

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]

        result = self.encoder(x)
        if isinstance(result, list):
            result = result[-1]
        return result

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss}
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        # result = pl.EvalResult(checkpoint_on=loss)
        self.log('avg_val_loss', loss)
        return loss
    
      

    def shared_step(self, batch, batch_idx):
        (img1, img2) = batch

        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 2048, 2, 2)
        h1 = self.encoder(img1).last_hidden_state
        h2 = self.encoder(img2).last_hidden_state
        # h1 = self.encoder(img1).logits
        # h2 = self.encoder(img2).logits

        # the bolts resnets return a list of feature maps
        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

        # PROJECT
        # img -> E -> h -> || -> z
        # (b, 2048, 2, 2) -> (b, 128)
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)

        return loss
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)