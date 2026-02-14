"""Training script for GoalFlow."""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goalflow.config import load_config
from goalflow.models.goal_flow_model import create_model
from goalflow.data import SimulatedGoalFlowDataset, create_dataloader


class GoalFlowLightningModule(pl.LightningModule):
    """PyTorch Lightning module for GoalFlow."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = create_model(config)

    def forward(self, batch):
        return self.model(
            images=batch["images"],
            lidar_points=batch.get("lidar_points"),
            ego_state=batch.get("ego_state"),
            ground_truth_goal=batch.get("goal"),
            ground_truth_trajectory=batch.get("trajectory"),
        )

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss_dict = self.model.compute_loss(
            outputs,
            ground_truth_goal=batch.get("goal"),
            ground_truth_trajectory=batch.get("trajectory"),
        )

        loss = loss_dict.get("loss_total", torch.tensor(0.0))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss_dict = self.model.compute_loss(
            outputs,
            ground_truth_goal=batch.get("goal"),
            ground_truth_trajectory=batch.get("trajectory"),
        )

        loss = loss_dict.get("loss_total", torch.tensor(0.0))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        training_config = self.config.get("training", {})

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 1e-4),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.get("max_epochs", 100),
        )

        return [optimizer], [scheduler]


def train(config_path: str = None):
    """Main training function."""
    # Load config
    if config_path is None:
        config = load_config()
    else:
        config = load_config(config_path)

    training_config = config.get("training", {})

    # Create data
    print("Creating dataset...")
    train_dataset = SimulatedGoalFlowDataset(
        num_samples=1000,
        num_cameras=config.get("data", {}).get("num_cameras", 3),
    )
    val_dataset = SimulatedGoalFlowDataset(
        num_samples=200,
        num_cameras=config.get("data", {}).get("num_cameras", 3),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.get("batch_size", 8),
        shuffle=True,
        num_workers=training_config.get("num_workers", 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.get("batch_size", 8),
        shuffle=False,
        num_workers=training_config.get("num_workers", 0),
    )

    # Create model
    print("Creating model...")
    model = GoalFlowLightningModule(config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="goalflow-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Logger
    logger = TensorBoardLogger("logs", name="goalflow")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=training_config.get("max_epochs", 100),
        accelerator=training_config.get("accelerator", "auto"),
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=training_config.get("gradient_clip", 1.0),
    )

    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    train(args.config)
