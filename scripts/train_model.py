import argparse

import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from datamodules import DataModuleFactory
from models import ModelFactory


def main(config: argparse.Namespace):
    print("Setting up data module and model...")
    model_factory = ModelFactory()
    model = model_factory.create_model(config)

    datamodule_factory = DataModuleFactory()
    dm = datamodule_factory.create_datamodule(config)

    # Run training
    print(f"Training {config.model} on {config.train_path} dataset for {config.task} task...")
    results_path = f"{config.model}_{config.task}"
    tb_logger = TensorBoardLogger(
        save_dir="results",
        name=results_path
    )
    csv_logger = CSVLogger(
        save_dir="results",
        name=results_path
    )
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

    # Run evaluation if configured
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
