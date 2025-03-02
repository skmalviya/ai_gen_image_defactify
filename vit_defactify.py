import os
import io
import random
from collections import OrderedDict

import torch
from PIL import Image
from datasets import load_dataset
from sympy.abc import sigma
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer

from torchvision import datasets
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


processor = ViTImageProcessor.from_pretrained('vit-base-patch16-224', do_rescale=False, return_tensors='pt')

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

train_transform = v2.Compose([
    # Random Resized Crop to add variability in size and perspective
    v2.RandomApply(transforms=[v2.ColorJitter(brightness=[0.45,0.55])], p=0.2),
    v2.RandomHorizontalFlip(0.2),
    v2.RandomVerticalFlip(0.1),
    v2.RandomApply(transforms=[v2.JPEG(quality=50)], p=0.2),
    v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 90))], p=0.2),
    v2.ToTensor(),
    v2.RandomApply(transforms=[v2.GaussianNoise(mean=0.0, sigma=0.3)], p=0.2),
    v2.RandomResizedCrop(size=(processor.size["height"], processor.size["width"]), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    # v2.Resize((processor.size["height"], processor.size["width"])),

    v2.Normalize(mean=image_mean, std=image_std)
    # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
])


test_transform = v2.Compose([
    v2.Resize((processor.size["height"], processor.size["width"])),
    v2.ToTensor(),
    v2.Normalize(mean=image_mean, std=image_std)
])

def train_transforms(image):
    return train_transform(image.convert("RGB"))

def test_transforms(image):
    return test_transform(image.convert("RGB"))

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return dict(f1=f1_score(predictions, labels))

def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


dataset_train = datasets.ImageFolder(root='/media/STORAGE/Shrikant/Defactify4.0/AI-image-detect/Defactify4_Train/saved_images', transform=train_transforms)
dataset_valid = datasets.ImageFolder(root='/media/STORAGE/Shrikant/Defactify4.0/AI-image-detect/Defactify4_Validation/saved_images', transform=test_transforms)

label2idx_defactify = dataset_train.class_to_idx
idx2label_defactify = {v: k for k, v in label2idx_defactify.items()}

model = ViTForImageClassification.from_pretrained('vit-base-patch16-224',
                                                  id2label=idx2label_defactify,
                                                  label2id=label2idx_defactify,
                                                  ignore_mismatched_sizes=True)
metric_name = "accuracy"

args = TrainingArguments(
    f"defactify-classification-aug_img_transform",
    use_cpu = False,
    evaluation_strategy="steps",
    logging_steps = 500,
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=256,
    num_train_epochs=15,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
    push_to_hub = False,
    # hub_model_id = "MyPetModel"
)

trainer = Trainer(
    model,
    args,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()
