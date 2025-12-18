import argparse

import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from datamodules import ImageFolderDataModule
from models import BatchNCNN, SimpleCNN, ResidualCNN, AdvancedBatchNCNN


def main(config: argparse.Namespace):
    print("Setting up data and model...")
    if config.task == "classification":
        if config.model == "SimpleCNN":
            model = SimpleCNN()
        elif config.model == "BatchCNN":
            model = BatchNCNN()
        elif config.model == "ResidualCNN":
            model = ResidualCNN()
        elif config.model == "AdvancedBatchCNN":
            model = AdvancedBatchNCNN()
        else:
            raise ValueError("Invalid model name.")

        # Prepare LightningDataModule
        dm = ImageFolderDataModule(
            train_dir=config.train_path,
            val_dir=config.val_path,
            test_dir=config.test_path,
            batch_size=8,
            num_workers=4,
            image_size=(128, 128),
            mean=(0.5,),
            std=(0.5,),
            augment=True,
        )
        dm.setup()

        # Run training and evaluation
        results_path = f"{config.model}_{config.task}"
        tb_logger = TensorBoardLogger(
            save_dir="results",
            name=results_path
        )
        csv_logger = CSVLogger(
            save_dir="results",
            name=results_path
        )
        print(f"Training {config.model} on {config.train_path} dataset for {config.task} task...")
        trainer = Trainer(
            max_epochs=config.epochs,
            logger=[tb_logger, csv_logger],
            callbacks=[
                EarlyStopping(monitor='val_acc', patience=3, mode='max'),
                ModelCheckpoint(
                    monitor='val_acc',
                    mode='max',
                    dirpath=f"results/{results_path}/version_0/checkpoints"
                )
            ]
        )
        trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
        print(f"Training done. Model saved to {config.save_path}.")
        if config.evaluation:
            print("Evaluating model on test set...")
            trainer.test(model, dm.test_dataloader())
            print("Evaluation done.")
        else:
            print("Evaluation skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert dictionary to Namespace
    current_config = argparse.Namespace(**config_dict)

    main(current_config)
