import sys  
sys.path.insert(0, '../')
import torch
import torch.nn as nn

from tools.datasets import ImageNet9
from tools.model_utils import make_and_restore_model, eval_model
from tools.folder import pil_loader
import tqdm
from matplotlib import pyplot as plt


n_classes = 9
def l_2_onehot(labels,nb_digits=n_classes):
    # take labels (from the dataloader) and return labels onehot-encoded
    label_onehot = torch.FloatTensor(labels.shape[0], nb_digits)
    label_onehot.zero_()
    label_onehot = label_onehot.scatter_(1,labels.unsqueeze(1),1).cpu()

    return label_onehot


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

def make_epoch(train_loader, cuda, optimizer, net, criterion, loss_train, acc_train, test_loader, loss_test, acc_test, epoch, test_acc_period, _print) :
  for data in train_loader:
    inputs, labels = data
    if cuda:
      inputs = inputs.type(torch.cuda.FloatTensor)
      labels = labels.type(torch.cuda.LongTensor)

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


def train(net, train_loader, test_loader,  n_epoch_first_train, test_acc_period = 5, cuda=True, criterion = nn.CrossEntropyLoss(), _print = True, initial_lr = 1e-4):
  loss_train = []
  loss_test = []
  acc_train = []
  acc_test = []
  lr_curve = []

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

  learning_rate = initial_lr
  lr_curve.append(learning_rate)

  # First training (lr cst)
  for epoch in tqdm.tqdm_notebook(range(n_epoch_first_train)):
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    make_epoch(train_loader, cuda, optimizer, net, criterion, loss_train, acc_train, test_loader, loss_test, acc_test, epoch, test_acc_period, _print)
    lr_curve.append(learning_rate)

  # Second training (lr decr)
  for epoch in tqdm.tqdm_notebook(range(n_epoch_first_train)):
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    make_epoch(train_loader, cuda, optimizer, net, criterion, loss_train, acc_train, test_loader, loss_test, acc_test, epoch, test_acc_period, _print)
    learning_rate *= 0.9
    lr_curve.append(learning_rate)

  print('Finished Training')
  return loss_train, loss_test, acc_train, acc_test, lr_curve





def make_base_training(variation, net, n_epoch_first_train, batch_size = 16, workers = 0, criterion = nn.CrossEntropyLoss(), test_acc_period = 5, _print = True, initial_lr = 1e-4) :
  # Creation of the datasets
  dataset = ImageNet9("../data/"+variation)
  val_loader = dataset.make_loaders(batch_size=batch_size, workers=workers)
  train_loader = dataset.make_loaders(batch_size=batch_size, workers=workers, shuffle_val=True, test = False)

  use_cuda = True
  if use_cuda and torch.cuda.is_available():
      print("using cuda")
      net.cuda()

  # Lauching the training
  loss_train, loss_test, acc_train, acc_test, lr_curve =  train(net = net, 
                                                                train_loader = train_loader,
                                                                test_loader = val_loader,
                                                                n_epoch_first_train = n_epoch_first_train,
                                                                test_acc_period = test_acc_period,
                                                                criterion = criterion,
                                                                _print = _print,
                                                                initial_lr = initial_lr
                                                                )
  
 

  acc, loss = accuracy(net, val_loader, cuda=use_cuda, criterion = nn.CrossEntropyLoss())

  # Display of the results
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

  print()
  print("Learning Rate Curve")
  plt.plot(range(len(lr_curve)), lr_curve, label = "learning rate")
  plt.legend()
  plt.show()



def test_on_dataset(variation, net, batch_size = 16, workers = 0) : 
    dataset = ImageNet9("../data/"+variation)
    val_loader = dataset.make_loaders(batch_size=batch_size, workers=workers)
    acc, loss = accuracy(net, val_loader, cuda=True)
    return acc