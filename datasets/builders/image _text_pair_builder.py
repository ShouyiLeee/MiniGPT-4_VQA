import logging

@registry.register_builder("vivqa")
class ViVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = ViVQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vivqa/default.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = {}

        # Create datasets
        dataset_cls = self.train_dataset_cls
        datasets["train"] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=build_info.image_path,
            ann_path=build_info.ann_path,
        )
        return datasets
    
    
@registry.register_builder("mvtec_ad")
class MVTECADBuilder(BaseDatasetBuilder):
    train_dataset_cls = MVTecDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mvtec/default.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # Create datasets
        dataset_cls = self.train_dataset_cls
        datasets["train"] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )
        return datasets


