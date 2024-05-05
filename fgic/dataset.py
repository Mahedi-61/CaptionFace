from torch.utils.data import Dataset,DataLoader
import os, random
from PIL import Image
import torch 
import torchvision.transforms as transforms


def get_images_labels(filename):
    fh = open(filename, 'r')
    imgs = []
    labels = []
    captions = []

    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        imgs.append(os.path.join("./data/cub/images", words[0])) 
        labels.append( int(words[1]) )
        captions.append(os.path.join("./data/cub/text", words[0].replace(".jpg", ".txt")))

    return imgs, labels, captions


def get_transform(train):
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
    ])

    if train == True: return transform_train
    elif train == False: return transform_test 



class CUBDataset(Dataset):
    def __init__(self, train, args):
        self.train = train
        if self.train == True: 
            filename = os.path.join(args.train_imgs_list)
            
        elif self.train == False:
            filename = os.path.join(args.test_imgs_list)

        self.imgs, self.labels, self.captions = get_images_labels(filename) 
        self.transform = get_transform(self.train)
        self.args = args

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        img = Image.open(img).convert('RGB')
        img = self.transform(img)
        
        # random select a sentence
        if self.train == True:
            sent_ix = random.randint(0, self.args.captions_per_image - 1)
        elif self.train == False:
            sent_ix = 0 #for consistent resutls 
            
        with open(self.captions[index], "r") as f:
            texts = f.read().splitlines()

        return img, torch.tensor(label), texts[sent_ix]

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    import types 
    args = types.SimpleNamespace()
    args.captions_per_image = 10
    args.train_imgs_list = "./data/cub/train_images_shuffle.txt"
    cub = CUBDataset(train=True, args=args)
