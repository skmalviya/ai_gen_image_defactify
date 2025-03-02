import os
import random
import json
import pandas as pd

import torch
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer

from torchvision import datasets
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    dict_acc_labels = dict(accuracy=accuracy_score(predictions, labels))
    dict_acc_labels['predictions'] = predictions
    dict_acc_labels['labels'] = labels
    return dict_acc_labels

def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

pet_dataset = load_dataset("PetClassification")

labels = pet_dataset["train"].features["label"].names

idx2label = {idx: label for idx, label in enumerate(labels)}
label2idx = {label: idx for idx, label in enumerate(labels)}

random_idx = random.randint(0, len(pet_dataset['train']))
print(f"Breed: {idx2label[pet_dataset['train'][random_idx]['label']]}")
pet_dataset['train'][random_idx]['image']



processor = ViTImageProcessor.from_pretrained('vit-base-patch16-224', do_rescale=False, return_tensors='pt')

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = v2.Normalize(mean=image_mean, std=image_std)

train_transform = v2.Compose([
    v2.Resize((processor.size["height"], processor.size["width"])),
    v2.RandomHorizontalFlip(0.4),
    v2.RandomVerticalFlip(0.1),
    v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 90))], p=0.5),
    v2.RandomApply(transforms=[v2.ColorJitter(brightness=.3, hue=.1)], p=0.3),
    v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.3),
    v2.ToTensor(),
    normalize
    # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
])

test_transform = v2.Compose([
    v2.Resize((processor.size["height"], processor.size["width"])),
    v2.ToTensor(),
    normalize
])


def train_transforms(image):
    return train_transform(image.convert("RGB"))

def test_transforms(image):
    return test_transform(image.convert("RGB"))

dataset_train = datasets.ImageFolder(root='/media/STORAGE/Shrikant/Defactify4.0/AI-image-detect/Defactify4_Train/saved_images', transform=train_transforms)

dataset_valid = datasets.ImageFolder(root='/media/STORAGE/Shrikant/Defactify4.0/AI-image-detect/Defactify4_Validation/saved_images', transform=test_transforms)

dataset_test = datasets.ImageFolder(root='/media/STORAGE/Shrikant/Defactify4.0/AI-image-detect/Defactify4_Test/saved_images', transform=test_transforms)

# Set the transforms
# pet_dataset['train'].set_transform(train_transforms)
# pet_dataset['validation'].set_transform(test_transforms)
# pet_dataset['test'].set_transform(test_transforms)

model = ViTForImageClassification.from_pretrained("breed-classification/checkpoint-9855",
                                                  id2label=idx2label,
                                                  label2id=label2idx,
                                                  ignore_mismatched_sizes=True)

metric_name = "accuracy"

args = TrainingArguments(
    f"breed-classification",
    use_cpu = False,
    evaluation_strategy="steps",
    logging_steps = 100,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
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


# trainer.train()
{'coco_image': 0, 'dalle_image': 1, 'midjourney_image': 2, 'sd21_image': 3, 'sd3_image': 4, 'sdxl_image': 5}
# save answer.csv file as required by the shared task
def save_answer_csv(labels, dataset):
    pred_labels = labels['eval_predictions']
    true_labels = labels['eval_labels']

    pred_map = {0: 0, 1: 4, 2: 5, 3: 1, 4: 3, 5: 2}
    data_tuple = []
    # Update the data based on the conditions
    for img_lbl, pred in zip(dataset.imgs, pred_labels):
        img_index = img_lbl[0].split('/')[-1].split('.')[0].split('_')[-1]

        t_dict = {
            "index": int(img_index),
            "caption": "",
            "Label_A": 0 if not pred else 1,
            "Label_B": pred_map[pred]
        }
        data_tuple.append([int(img_index), t_dict])
    data_tuple = sorted(data_tuple, key=lambda x: x[0])
    pd_val = pd.read_excel('/media/STORAGE/Shrikant/Defactify4.0/AI-image-detect/Defactify4_Test/captions.xlsx')
    print(len(pd_val['Caption'].values))
    data = [e[1] for e in data_tuple]
    data1 = []
    for d, c in zip(data, pd_val['Caption'].values):
        d['caption'] = c
        data1.append(d)
    with open('output_test.csv', 'w') as f_csv:
        for d in data_tuple:
            f_csv.write(','.join([str(d[1]['index']), str(d[1]['Label_A']), str(d[1]['Label_B'])]).strip() + '\n')
    # Save the updated data to a JSON file
    with open('output_test.json', 'w') as json_file:
        json.dump(data1, json_file, indent=4)

#Valid
# acc_valid = trainer.evaluate(dataset_valid)
# save_answer_csv(acc_valid, dataset_valid)

#Test
acc_test = trainer.evaluate(dataset_test)
save_answer_csv(acc_test, dataset_test)

exit()

# Map predicted indices to class names
class_names = ["original", "AI_gen1", "AI_gen2", "AI_gen3", "AI_gen4", "AI_gen5"]
predicted_class_names = [class_names[label] for label in predicted_labels]

# Step 7: Save Predictions to CSV
results = pd.DataFrame({
    "image_path": [item["path"] for item in test_dataset],
    "predicted_label": predicted_class_names
})
results.to_csv("predictions_with_checkpoint.csv", index=False)
print("Predictions saved to predictions_with_checkpoint.csv")

s = 1
