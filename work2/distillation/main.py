import torch
from torch import nn, optim
import argparse
import numpy as np
import random
from models import resnet, TeacherNet
from torch.utils import data
from dataloader import HDF5Dataset
from train import validate_epoch, train_epoch
import os

# set flags
parser = argparse.ArgumentParser(description='PACS')
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=1)
flags = parser.parse_args()

# set seed
random.seed(flags.seed)

# load train data
dataset1 = HDF5Dataset('/opt/data/private/TransDG/SDViT/domainbed/data/PACS/art_painting')
dataset2 = HDF5Dataset('/opt/data/private/TransDG/SDViT/domainbed/data/PACS/cartoon')
dataset3 = HDF5Dataset('/opt/data/private/TransDG/SDViT/domainbed/data/PACS/photo')
train_data = data.DataLoader(data.ConcatDataset([dataset1, dataset2, dataset3]), num_workers=1, 
                                                batch_size=flags.batch_size, shuffle=True, drop_last=True)

# load test data
dataset4 = HDF5Dataset('/opt/data/private/TransDG/SDViT/domainbed/data/PACS/sketch')
test_data = data.DataLoader(dataset4, num_workers=1, batch_size=flags.batch_size, 
                              shuffle=True, drop_last=True)

# load model
model = resnet(hidden_dim=flags.hidden_dim, num_classes=7).cuda()
# if os.path.exists("/opt/data/private/TransDG/liuyangcen/teacher.pth"):
#    model.load_state_dict(torch.load("/opt/data/private/TransDG/liuyangcen/teacher.pth"))

teacher_model = TeacherNet("/opt/data/private/TransDG/liuyangcen/fft_train.pth")

# set train function 
def trainer(model, train_data, test_data, epochs, learning_rate, teacher_model):
    # set loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # loop over the dataset multiple times
    for epoch in range(epochs):
        # train one epoch
        train_epoch(train_data, model, loss_function, optimizer, epoch, epochs, teacher_model)
        # validate epoch on validation set
        loss_train, accuracy_train, loss_test, accuracy_test = validate_epoch(train_data, test_data, model, loss_function)
        # print the metrics
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                                np.array2string(loss_train, precision=2, floatmode='fixed'),
                                np.array2string(accuracy_train, precision=2, floatmode='fixed'),
                                np.array2string(loss_test, precision=2, floatmode='fixed'),
                                np.array2string(accuracy_test, precision=2, floatmode='fixed')))          
        torch.save(model.state_dict(), '/opt/data/private/TransDG/liuyangcen/teacher.pth')
        print('Save Checkpoint')
    print('Finished Training')


if __name__ == "__main__":
    trainer(model, train_data, test_data, flags.epochs, flags.lr, teacher_model)
