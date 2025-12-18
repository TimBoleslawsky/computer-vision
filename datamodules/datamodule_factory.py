from .geotif_datamodule import GeoTiffDataModule
from .image_folder_datamodule import ImageFolderDataModule


class DataModuleFactory:
    @staticmethod
    def create_datamodule(config):
        if config.datamodule == "ImageFolder":
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
        elif config.datamodule == "GeoTiff":
            dm = GeoTiffDataModule()
        else:
            raise ValueError("Invalid datamodule name.")

        dm.setup()
        return dm
