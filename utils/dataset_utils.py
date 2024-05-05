from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import _pickle as pickle
import gc
from transformers import  AutoTokenizer, CLIPTokenizer
from albumentations.pytorch import ToTensorV2
import albumentations as A 
from utils import attribute as a 


def encode_Bert_tokens(text_encoder, text_head, caption, mask, args):
    caption = Variable(caption).to(args.device)
    mask = Variable(mask).to(args.device)

    with torch.no_grad():
        words_emb, sent_emb_org = text_encoder(caption, mask)
        words_emb, sent_emb = text_head(words_emb, sent_emb_org)

    return words_emb.detach(), sent_emb.detach()


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap


def get_imgs(img_path, split, model_type="arcface"):

    img = np.array(Image.open(img_path).convert('RGB')) 
    sample_transforms = [
        A.HorizontalFlip(),
        A.ColorJitter(),
        A.RandomBrightnessContrast(),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=30, p=0.5),
        A.HueSaturationValue(p=0.3)
    ]

    train_transforms = A.Compose([
        *sample_transforms,
        A.Normalize(mean = [0.5, 0.5, 0.5], 
                    std=[0.5, 0.5, 0.5],
                    always_apply=True),
        ToTensorV2()
    ])

    valid_transforms = A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
        ToTensorV2()
    ])

    if split == "train": tfms = train_transforms
    elif split == "test" or split == "valid":  tfms = valid_transforms

    img = tfms(image=img)["image"] 
    if model_type == "adaface": 
        permute = [2, 1, 0]
        img = img[permute, :, :] #RGB --> BGR

    return img


def do_flip_test_images(img_path, model_type="arcface"):
    img = np.array(Image.open(img_path).convert('RGB')) 
    tfms = A.Compose([
        A.HorizontalFlip(p = 1),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
        ToTensorV2()
    ])

    img = tfms(image=img)["image"] 
    if model_type == "adaface": 
        permute = [2, 1, 0]
        img = img[permute, :, :] #RGB --> BGR
    return img


def load_captions_Bert(data_dir, filenames, args):
    # convert the raw text into a list of tokens.
    # attention_mask (which tokens should be used by the model 1 - use or 0 - don’t use).
    if args.bert_type == "bert": 
        tokenizer = AutoTokenizer.from_pretrained(args.bert_config)

    elif args.bert_type == "align":
        tokenizer = AutoTokenizer.from_pretrained(args.align_config, use_fast=False)

    elif args.bert_type == "clip":
         tokenizer = AutoTokenizer.from_pretrained(args.clip_config, use_fast=False)

    elif args.bert_type == "blip":
         tokenizer = AutoTokenizer.from_pretrained(args.blip_config, use_fast=False)


    all_captions = []
    all_attention_mask = []
    all_attr_label = []

    for i in range(len(filenames)):
        cap_path = os.path.join(data_dir, "text", filenames[i] + ".txt")
    
        with open(cap_path, "r") as f:
            captions = f.read().encode('utf-8').decode('utf8').split('\n')
            cnt = 0
            for cap in captions:
                if len(cap) == 0: continue
                cap = cap.replace("\ufffd\ufffd", " ")
                
                encoding = tokenizer.encode_plus(
                            cap,
                            add_special_tokens=True,
                            max_length = args.bert_words_num,
                            return_token_type_ids=False,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt')

                input_ids=encoding["input_ids"].flatten()
                attention_mask=encoding["attention_mask"].flatten()

                all_captions.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_attr_label.append(a.get_attr_vector(cap))

                cnt += 1
                if cnt == args.captions_per_image:
                    break

            if cnt < args.captions_per_image:
                print('ERROR: the captions for %s less than %d' % (filenames[i], cnt))

    del captions 
    return all_captions, all_attention_mask, all_attr_label



def load_captions(data_dir, filenames, embeddings_num):
    all_captions = []
    for i in range(len(filenames)):
        cap_path = os.path.join(data_dir, "text", filenames[i] + ".txt")
    
        with open(cap_path, "r") as f:
            captions = f.read().encode('utf-8').decode('utf8').split('\n')
            cnt = 0
            for cap in captions:
                if len(cap) == 0: continue
                cap = cap.replace("\ufffd\ufffd", " ")

                # picks out sequences of alphanumeric characters as tokens and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                
                if len(tokens) == 0:
                    print('cap', cap)
                    continue
                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                
                all_captions.append(tokens_new)
                cnt += 1
                if cnt == embeddings_num:
                    break

            if cnt < embeddings_num:
                print('ERROR: the captions for %s less than %d' % (filenames[i], cnt))

    del captions 
    return all_captions



def load_text_data_Bert(data_dir, args):
    filepath = os.path.join(data_dir, 'captions_%s.pickle' % args.bert_type)

    if not os.path.isfile(filepath):
        train_names = load_filenames(data_dir, 'train')
        valid_names = load_filenames(data_dir, 'valid')
        test_names = load_filenames(data_dir, 'test')

        train_captions, train_att_masks, train_attr_label = load_captions_Bert(data_dir, train_names, args)
        valid_captions, valid_att_masks, valid_attr_label = load_captions_Bert(data_dir, valid_names, args)
        test_captions, test_att_masks, test_attr_label = load_captions_Bert(data_dir, test_names, args)

        with open(filepath, 'wb') as f:
            pickle.dump([train_captions, train_att_masks, train_attr_label, 
                         valid_captions, valid_att_masks, valid_attr_label,
                         test_captions, test_att_masks, test_attr_label], 
                        f, protocol=2)
            print('\nSave to: ', filepath)
    else:
        print("Loading ", filepath)
        with open(filepath, 'rb') as f:
            gc.disable()
            x = pickle.load(f)
            gc.enable()
            train_captions, train_att_masks, train_attr_label, valid_captions, valid_att_masks, valid_attr_label,\
                test_captions, test_att_masks, test_attr_label = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]  ######fix here and delete the bert caption
        del x

    train_names = load_filenames(data_dir, 'train')
    valid_names = load_filenames(data_dir, 'valid')
    test_names = load_filenames(data_dir, 'test')

    print("loading complete")
    return (train_names, train_captions, train_att_masks, train_attr_label,
            valid_names, valid_captions, valid_att_masks, valid_attr_label,
            test_names, test_captions, test_att_masks, test_attr_label) 


def load_filenames(data_dir, split):
    filepath = os.path.join(data_dir, split, "filenames.pickle")

    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load %s filenames from: %s (%d)' % (split, filepath, len(filenames)))
        #print("Sample %s filenames: %s" % (split, filenames[0]))
    else:
        filenames = []
    return filenames


def load_class_id(data_dir):
    filepath = os.path.join(data_dir, "class_info.pickle")

    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            gc.disable()
            class_id = pickle.load(f, encoding="bytes")
            gc.enable()

    print('Load class_info from: %s (%d)' % (filepath, len(class_id)))
    return class_id


all_attributes = ["5_o_Clock_Shadow",	"Arched_Eyebrows",	"Attractive",	"Bags_Under_Eyes",	"Bald",	
                "Bangs",	"Big_Lips",	"Big_Nose",	"Black_Hair", "Blond_Hair",	
                "Blurry",	"Brown_Hair",	"Bushy_Eyebrows",	"Chubby",	"Double_Chin",
                "Eyeglasses",	"Goatee",	"Gray_Hair",	"Heavy_Makeup",	"High_Cheekbones",
                "Male",	"Mouth_Slightly_Open",	"Mustache", "Narrow_Eyes", "No_Beard",
                "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",	
                "Sideburns", "Smiling",	"Straight_Hair", 	"Wavy_Hair",	"Wearing_Earrings",
                "Wearing_Hat",	"Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]