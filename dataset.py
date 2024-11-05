import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image


class NATURE(Dataset):
    def __init__(self, root, is_train = True, is_test = False, transform = None):
        self.root = os.path.join(root, 'train') if is_train else os.path.join(root,'valid')
        self.features = os.listdir(self.root)
        self.images = []
        self.labels = []
        for index,feature in enumerate(self.features):
            images = os.listdir(os.path.join(self.root, feature))
            for image in images:
                self.images.append(os.path.join(os.path.join(self.root, feature), image))
                self.labels.append(index)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label

if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224,224))
    ])
    dataset = NATURE(root="Dataset", is_train=True, transform = transform)
    image,label=dataset.__getitem__(100)

