from torch.utils.data import Dataset,DataLoader
import os 
from PIL import Image
import torch 
import torchvision.transforms as transforms


def get_images_labels(filename):
    fh = open(filename, 'r')
    imgs = []
    labels = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        imgs.append(os.path.join("./data/cub/images", words[0])) 
        labels.append( int(words[1]) )

    return imgs, labels


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
    def __init__(self, train = True):

        if train == True: 
            filename = os.path.join("./data/cub/train_images_shuffle.txt")
        elif train == False:
            filename = os.path.join("./data/cub/test_images_shuffle.txt")

        self.imgs, self.labels = get_images_labels(filename) 
        self.transform = get_transform(train)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        img = Image.open(img).convert('RGB')
        img = self.transform(img)

        return img, torch.tensor(label)

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    cub = CUBDataset(train=False) 
    img, label = cub.__getitem__(0)
    print(img.shape)
    print(label)
