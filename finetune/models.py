from torch import optim
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from timm.optim import create_optimizer_v2
from dpt.models import DPTDepthModel
from .losses import SILog
from .datasets import Nutrition5k


class DPTModule(LightningModule):
    def __init__(
        self,
        model_path,
        dataset_path,
        scale=0.0000305,
        shift=0.1378,
        batch_size=16,
        base_lr=1e-5,
        max_lr=1e-4,
        num_workers=2,
        image_size=(384, 384),
        **kwargs
    ):

        super().__init__(**kwargs)

        self.model = DPTDepthModel(
            path=model_path,
            scale=scale,
            shift=shift,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        self.batch_size = batch_size
        self.base_lr = base_lr
        self.max_lr = max_lr

        self._dataset_path = dataset_path
        self._num_workers = num_workers
        self._image_size = image_size

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
            step_size_up=4 * len(self._nutrition5k_train) // self.batch_size,
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
            self._nutrition5k_train = Nutrition5k(
                "train",
                self._dataset_path,
                self._image_size,
            )
            self._nutrition5k_val = Nutrition5k(
                "val",
                self._dataset_path,
                self._image_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self._nutrition5k_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._nutrition5k_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
        )
