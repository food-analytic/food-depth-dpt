from math import ceil
from torch import optim
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from timm import create_optimizer_v2
from ..dpt.models import DPTDepthModel
from .loss import SILog
from .dataset import Nutrition5k


class DPTModule(LightningModule):
    def __init__(
        self,
        model_path: str,
        batch_size: int,
        base_lr: float,
        max_lr: float,
        base_scale: float = 1.0,
        base_shift: float = 0.0,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.model = DPTDepthModel(
            path=model_path,
            scale=base_scale,
            shift=base_shift,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        self.batch_size = batch_size
        self.base_lr = base_lr
        self.max_lr = max_lr

        self._loss_function = SILog

        self.model.pretrained.requires_grad_(False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        mask = mask.bool()
        yhat = self.model(x)
        loss = self._loss_function(yhat, y, mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        mask = mask.bool()
        yhat = self.model(x)
        loss = self._loss_function(yhat, y, mask)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            "madgrad",
        )
        cyclic_lr = optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=self.base_lr,
            max_lr=self.max_lr,
            step_size_up=4 * ceil(len(self._nutrition5k_train) / self.batch_size),
            cycle_momentum=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": cyclic_lr,
                "interval": "step",
            },
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self._nutrition5k_train = Nutrition5k(is_train=True)
            self._nutrition5k_val = Nutrition5k(is_train=False)

    def train_dataloader(self):
        return DataLoader(
            self._nutrition5k_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._nutrition5k_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
