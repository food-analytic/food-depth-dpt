from finetune import DPTModule
from configs import get_train_parser
import pytorch_lightning as pl

if __name__ == "__main__":

    args = get_train_parser().parse_args()

    with open(args.excluded_files, "r") as infile:
        excluded_files = [line.strip() for line in infile]

    model = DPTModule(
        args.model_path,
        args.dataset_path,
        scale=args.scale,
        shift=args.shift,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        num_workers=args.num_workers,
        image_size=args.image_size,
        excluded_files=excluded_files,
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=args.save_log)
    callbacks = []
    callbacks.append(pl.callbacks.LearningRateMonitor())
    callbacks.append(pl.callbacks.ModelCheckpoint(args.save_ckpt))
    if args.early_stopping is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping)
        )

    trainer = pl.Trainer(
        devices=args.devices,
        accelerator=args.accelerator,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, ckpt_path=args.load_ckpt)
