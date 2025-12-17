import argparse

import yaml

from src.models import BatchNCNN, SimpleCNN, ResidualCNN, AdvancedBatchNCNN
from src.processors import CustomDataLoader
from src.runners import Runner


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

        data_loader = CustomDataLoader
        train_loader, val_loader, test_loader = data_loader.load_data(
            config.train_path,
            config.validation_path,
            config.test_path
        )

        print(f"Training {config.model} on {config.train_path} dataset for {config.task} task...")
        runner = Runner(model)
        runner.run_training(train_loader, val_loader, config.save_path)
        print(f"Training done. Model saved to {config.save_path}.")
        if config.evaluation:
            print("Evaluating model on test set...")
            runner.run_evaluation(test_loader, config.save_path)
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
