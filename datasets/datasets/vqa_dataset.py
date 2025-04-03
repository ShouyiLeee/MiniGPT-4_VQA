import torch
from datasets.base_dataset import BaseDataset
import os
import random
import pandas as pd
import torch
from PIL import Image


class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


class ViVQADataset(torch.utils.data.Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]
        
        self.data = pd.read_csv(ann_path)
        self.img_name_prefix = "COCO_train2014_"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image_id = str(sample["img_id"]).zfill(12)
        image_path = os.path.join(self.vis_root, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(sample["question"])
        answer = self.text_processor(sample["answer"])
        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img> <ImageHere> </Img> {}".format(instruction)
        
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": image_id
        }

