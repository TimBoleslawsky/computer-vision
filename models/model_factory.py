from .cnns import SimpleCNN, SimpleCNNSegmenter, BatchNCNN, ResidualCNN, AdvancedBatchNCNN


class ModelFactory:
    @staticmethod
    def create_model(config):
        if config.model == "SimpleCNN":
            model = SimpleCNN()
        elif config.model == "BatchCNN":
            model = BatchNCNN()
        elif config.model == "ResidualCNN":
            model = ResidualCNN()
        elif config.model == "AdvancedBatchCNN":
            model = AdvancedBatchNCNN()
        elif config.model == "SimpleCNNSegmenter":
            model = SimpleCNNSegmenter(num_classes=config.num_classes)
        else:
            raise ValueError("Invalid model name.")

        return model
