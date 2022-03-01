import sys  
sys.path.insert(0, '../')
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
from argparse import ArgumentParser
from tools.datasets import ImageNet, ImageNet9
from tools.model_utils import make_and_restore_model, eval_model
from tools.folder import pil_loader
from PIL import Image
import tqdm
from matplotlib import pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


n_classes = 9
def l_2_onehot(labels,nb_digits=n_classes):
    # take labels (from the dataloader) and return labels onehot-encoded
    #
    # your code here
    #
    label_onehot = torch.FloatTensor(labels.shape[0], nb_digits)
    label_onehot.zero_()
    label_onehot = label_onehot.scatter_(1,labels.unsqueeze(1),1).cpu()

    return label_onehot




def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)



def accuracy(net, test_loader, cuda=True, criterion = nn.CrossEntropyLoss()):
  net.eval()
  correct = 0
  total = 0
  loss = 0
  with torch.no_grad():
      for data in test_loader:
          images, labels = data
          if cuda:
            images = images.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.LongTensor)
          outputs = net(images)
          
          loss += criterion(outputs, labels)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  net.train()

  return 100.0 * correct/ total, loss.item()


def train(net, optimizer, train_loader, test_loader, loss,  n_epoch = 5, test_acc_period = 5, cuda=True, criterion = nn.CrossEntropyLoss(), _print = True):

  initialize_weights(net)

  loss_train = []
  loss_test = []
  acc_train = []
  acc_test = []

  train_acc, train_loss = accuracy(net, train_loader, cuda=cuda, criterion = criterion)
  loss_train.append(train_loss)
  acc_train.append(train_acc)

  test_acc, test_loss = accuracy(net, test_loader, cuda=cuda, criterion = criterion)
  loss_test.append(test_loss)
  acc_test.append(test_acc)
  if _print :
    print('[%d] train loss: %.3f' %(0, train_loss))
    print('[%d] train acc: %.3f' %(0, train_acc))
    print()
    print('[%d] test loss: %.3f' %(0, test_loss))
    print('[%d] test acc: %.3f' %(0, test_acc))
    print("####################")

  for epoch in tqdm.tqdm_notebook(range(n_epoch)):  # loop over the dataset multiple times

    for data in train_loader:
      # get the inputs
      inputs, labels = data
      if cuda:
        inputs = inputs.type(torch.cuda.FloatTensor)
        labels = labels.type(torch.cuda.LongTensor)
      # print(inputs.shape)

      # zero the parameter gradients
      optimizer.zero_grad()

      outputs = net(inputs)

      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    train_acc, train_loss = accuracy(net, train_loader, cuda=cuda, criterion = criterion)
    loss_train.append(train_loss)
    acc_train.append(train_acc)

    test_acc, test_loss = accuracy(net, test_loader, cuda=cuda, criterion = criterion)
    loss_test.append(test_loss)
    acc_test.append(test_acc)
    
    if ((epoch+1) % test_acc_period == 0) and _print:
      print('[%d] train loss: %.3f' %(epoch + 1, train_loss))
      print('[%d] train acc: %.3f' %(epoch + 1, train_acc))
      print()
      print('[%d] test loss: %.3f' %(epoch + 1, test_loss))
      print('[%d] test acc: %.3f' %(epoch + 1, test_acc))
      print("####################")
    
  print('Finished Training')
  return loss_train, loss_test, acc_train, acc_test

def make_training(variation, net, nb_epoch = 30, batch_size = 16, workers = 0, criterion = nn.CrossEntropyLoss(), test_acc_period = 5, _print = True) :

    dataset = ImageNet9("../data/"+variation)
    val_loader = dataset.make_loaders(batch_size=batch_size, workers=workers)
    train_loader = dataset.make_loaders(batch_size=batch_size, workers=workers, shuffle_val=True, test = False)


    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print("using cuda")
        net.cuda()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    loss_train, loss_test, acc_train, acc_test = train(net,
                                                      optimizer,
                                                      train_loader,
                                                      val_loader,
                                                      criterion,
                                                      n_epoch = nb_epoch,
                                                      test_acc_period = test_acc_period,
                                                      criterion = nn.CrossEntropyLoss(),
                                                      _print = _print)

    acc, loss = accuracy(net, val_loader, cuda=use_cuda, criterion = nn.CrossEntropyLoss())

    print("Final acc : ", acc)
    print()
    print("Accuracy Graph")
    plt.plot(range(len(acc_train)), acc_train, label = "train")
    plt.plot(range(len(acc_test)), acc_test, label = "test")
    plt.legend()
    plt.show()

    print()
    print("Loss Graph")
    plt.plot(range(len(loss_train)), loss_train, label = "train")
    plt.plot(range(len(loss_test)), loss_test, label = "test")
    plt.legend()
    plt.show()

def test_on_dataset(variation, net, batch_size = 16, workers = 0) : 
    dataset = ImageNet9("../data/"+variation)
    val_loader = dataset.make_loaders(batch_size=batch_size, workers=workers)
    acc, loss = accuracy(net, val_loader, cuda=True)
    return acc