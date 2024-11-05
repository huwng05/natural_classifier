import argparse
import os.path
from os import makedirs

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToTensor, Resize, Normalize
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import shutil
from dataset import NATURE
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def get_args():
    parser = argparse.ArgumentParser("Train arguments")
    parser.add_argument('--image-size','-i', type=int, default=224)
    parser.add_argument('--epoch','-e',type=int, default=100)
    parser.add_argument('--batch-size', '-b', type=int, default=64)
    parser.add_argument('--num-workers','-w',type=int, default=10)
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-2)
    parser.add_argument('--momentum', '-m', type=float, default=0.9)
    parser.add_argument('--dataset-root', '-d', type=str, default='Dataset')
    parser.add_argument('--save-path','-s',type=str, default='model')
    parser.add_argument('--summary-path', '-u', type=str, default='summary/dataset')
    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.isdir(args.summary_path):
        shutil.rmtree(args.summary_path)
    os.makedirs(args.summary_path)
    writer = SummaryWriter(args.summary_path)


    transform_train = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    parm = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": True,
        "drop_last": False
    }
    data_train = NATURE(args.dataset_root, is_train=True, transform=transform_train)
    dataloader_train = DataLoader(data_train,**parm)

    data_valid = NATURE(args.dataset_root, is_train=False, transform=transform_train)
    dataloader_valid = DataLoader(data_valid,**parm)

    # Model
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    model.fc = nn.Linear(in_features=2048, out_features=len(data_train.features),bias=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    if os.path.isdir(args.save_path):
        save = torch.load(os.path.join(args.save_path,'last.pt'))
        model.load_state_dict(save['model'])
        optimizer.load_state_dict(save['optimizer'])
        start_epoch = save['epoch']
        best_score = save['best']
    else:
        os.makedirs(args.save_path)
        start_epoch = 0
        best_score = -1

    for epoch in range(start_epoch, args.epoch):
        # Train model
        model.train()
        process_bar = tqdm.tqdm(dataloader_train, colour='blue')
        list_loss = []
        for iter, (image, label) in enumerate(process_bar):
            image = image.to(device)
            label = label.to(device)
            prediction = model(image)
            loss = criterion(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            list_loss.append(loss.item())
            optimizer.step()
            process_bar.set_description('Epoch: {}/{}, Loss: {:0.4f}'.format(epoch+1, args.epoch, np.mean(list_loss)))
            writer.add_scalar("Train/Loss", np.mean(list_loss), epoch*args.epoch+iter)

        # Model Valid
        model.eval()
        with torch.no_grad():
            process_bar = tqdm.tqdm(dataloader_valid)
            list_loss = []
            list_label = []
            list_pre = []
            for iter, (image, label) in enumerate(process_bar):
                image = image.to(device)
                label = label.to(device)
                prediction = model(image)
                loss = criterion(prediction, label)
                list_loss.append(loss.item())
                process_bar.set_description('Epoch: {}, Loss: {}'.format(epoch + 1, np.mean(list_loss)))
                list_label.extend(label.tolist())
                prediction_classes = torch.argmax(prediction, dim=1)
                list_pre.extend(prediction_classes.tolist())
            acc = accuracy_score(list_label,list_pre)
            print("Epoch: {}, Accuracy: {:0.5f}, Loss: {:0.4f}".format(epoch+1,acc,np.mean(list_loss)))
            writer.add_scalar("Val/Loss", np.mean(list_loss), epoch+1)
            writer.add_scalar("Val/Accuracy", acc)
            conf_matrix = confusion_matrix(list_label, list_pre)
            plot_confusion_matrix(writer, conf_matrix ,[i for i in range(len(data_train.features))], epoch+1)

            checkpoint = {
                'epoch': epoch+1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best': best_score
            }

            torch.save(checkpoint, os.path.join(args.save_path, "last.pt"))
            if acc > best_score:
                best_score = acc
                torch.save(checkpoint, os.path.join(args.save_path,"best.pt"))

if __name__ == '__main__':
    args = get_args()
    train(args)