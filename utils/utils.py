import os
import errno
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from albumentations.pytorch import ToTensorV2
from transformers import GPT2TokenizerFast
import albumentations as A 
from PIL import Image
from sklearn.model_selection import train_test_split 
import json 
from torch.utils.data import Dataset
import torch
import yaml
from easydict import EasyDict as edict
import datetime
import dateutil.tz
from datetime import date
today = date.today()

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# config
def get_time_stamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')  
    return timestamp


def load_yaml(filename):
    with open(filename, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg


def merge_args_yaml(args):
    if args.cfg_file is not None:
        opt = vars(args)
        args = load_yaml(args.cfg_file)
        args.update(opt)
        args = edict(args)
    return args


def save_args(save_path, args):
    fp = open(save_path, 'w')
    fp.write(yaml.dump(args))
    fp.close()



def load_model_weights(model, weights, train=True):
    multi_gpus = True 
    if list(weights.keys())[0].find('module')==-1:
        pretrained_with_multi_gpu = False
    else:
        pretrained_with_multi_gpu = True
    if (multi_gpus==False) or (train==False):
        if pretrained_with_multi_gpu:
            state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        else:
            state_dict = weights
    else:
        state_dict = weights
    model.load_state_dict(state_dict)
    return model



def load_models(net, metric_fc, optim, path):
    print("loading full tgfr model .....")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    net = load_model_weights(net, checkpoint['model']['net'])
    metric_fc = load_model_weights(metric_fc, checkpoint['model']['metric_fc'])
    optim = optim.load_state_dict(checkpoint['optimizer']['optimizer'])
    return net, metric_fc, optim 


def load_fusion_net(net, path):
    print("loading fusion model .....")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    net = load_model_weights(net, checkpoint["net"])
    return net


def get_dataframes(dataset, train = True):
    base_path = os.path.join("./data", dataset, "annotations")

    if dataset == "coco2017":
        annot = os.path.join(base_path, "captions_train2017.json")
        with open(annot, "r") as f:
            data = json.load(f)
            data = data["annotations"]

        samples = []
        for sample in data:
            im = "%012d.jpg" % sample["image_id"]
            samples.append([im, sample["caption"]])
        
        df = pd.DataFrame(samples, columns=["image", "caption"])
        df["image"] = df["image"].apply(lambda x: os.path.join(base_path, "train2017", x))
        df = df.sample(150_000)
        

    elif dataset == "face2text" or dataset == "celeba":
        annot = os.path.join(base_path, "output.csv")
        with open(annot, "r") as f:
            df = pd.read_csv(f)

        df["image"] = df["image"].apply(lambda x: os.path.join(base_path, "train_images", x))


    df = df.reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.1)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    if train==True: return train_df
    elif train==False: return val_df  


def get_tfms(train = True):
    sample_tfms = [
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(),
        A.ColorJitter(),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
        A.HueSaturationValue(p=0.3)
    ]

    train_tfms = A.Compose([
        *sample_tfms,
        A.Resize(112, 112),
        A.Normalize(mean = [0.5, 0.5, 0.5], 
                    std=[0.5, 0.5, 0.5],
                    always_apply=True),
        ToTensorV2()
    ])

    valid_tfms = A.Compose([
        A.Resize(112, 112),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
        ToTensorV2()
    ])

    if train==True: return train_tfms
    elif train==False: return valid_tfms



class Img2CapDataset(Dataset):
    def __init__(self, dataset, train = True):
        self.tfms = get_tfms(train)
        self.df = get_dataframes(dataset, train)

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :]
        image = sample["image"]
        caption = sample["caption"]

        image = Image.open(image).convert("RGB")
        image = np.array(image)
        augs = self.tfms(image = image)        
        image = augs["image"]

        caption = f"{caption}<|endoftext|>"
        input_ids = self.tokenizer(caption, truncation=True)["input_ids"]
        labels = input_ids.copy()
        labels[:-1] = input_ids[1:]
        return image, input_ids, labels, sample["caption"]


def show_plot(df):
    sampled_df = df.sample(n=10)
    fig, axs = plt.subplots(5, 2, figsize=(20, 30))

    for i, row in enumerate(sampled_df.iterrows()):
        ax = axs[i // 2, i % 2]
        image_path = row[1]['image']
        caption = row[1]['caption']
        image = Image.open(image_path)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(caption)

    plt.tight_layout()
    plt.show()
