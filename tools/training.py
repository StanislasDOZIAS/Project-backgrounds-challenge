import sys  
sys.path.insert(0, '../')
import torch
import torch.nn as nn
import numpy as np
from tools.datasets import ImageNet9
import tqdm
from matplotlib import pyplot as plt


n_classes = 9
def l_2_onehot(labels,nb_digits=n_classes):
    # take labels (from the dataloader) and return labels onehot-encoded
    label_onehot = torch.FloatTensor(labels.shape[0], nb_digits)
    label_onehot.zero_()
    label_onehot = label_onehot.scatter_(1,labels.unsqueeze(1),1).cpu()

    return label_onehot

def polyak_update(polyak_factor, target_network, network):
    params1 = network.state_dict()
    params2 = target_network.state_dict()

    states = params2.copy()

    for name in states:
        states[name].data.copy_(polyak_factor*params1[name] + (1-polyak_factor)*params2[name].data)
    
    target_network.load_state_dict(states)

def accuracy(net, test_loader, cuda=True):
  net.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for data in test_loader:
          images, labels = data
          if cuda:
            images = images.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.LongTensor)
          outputs = net(images)
          
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  net.train()
  return 100.0 * correct/ total

def make_epoch(train_loader, cuda, optimizer, net, criterion) :
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


def make_epoch_no_back(train_loader, cuda, net) :
  for data in train_loader:
    inputs, labels = data
    if cuda:
      inputs = inputs.type(torch.cuda.FloatTensor)
      labels = labels.type(torch.cuda.LongTensor)
    net(inputs)


def train(net, temp_net, train_loader, original_val, mixed_same_val, mixed_rand_val, n_epoch_first_train, n_cycle, n_epoch_cycle , test_acc_period = 5, cuda=True, criterion = nn.CrossEntropyLoss(), _print = True, initial_lr = 1e-4):
  original_acc = []
  mixed_same_acc = []
  mixed_rand_acc = []
  learning_rate = initial_lr

  # In fact we go never here, but it's in case of total training in on shot.
  if n_epoch_first_train > 0 :
    # First training (lr cst) 
    for epoch in tqdm.tqdm_notebook(range(n_epoch_first_train)):
      optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
      make_epoch(train_loader, cuda, optimizer, net, criterion)

    # Second training (lr decr)
    for epoch in tqdm.tqdm_notebook(range(n_epoch_first_train)):
      optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
      make_epoch(train_loader, cuda, optimizer, net, criterion)
      learning_rate *= 0.9

  polyak_update(1, temp_net, net)

  original_acc.append(accuracy(temp_net, original_val, cuda=cuda))
  mixed_same_acc.append(accuracy(temp_net, mixed_same_val, cuda=cuda))
  mixed_rand_acc.append(accuracy(temp_net, mixed_rand_val, cuda=cuda))



  # Third training (lr cycle)
  for cycle in tqdm.tqdm_notebook(range(n_cycle)) :
    learning_rate /= np.power(0.9,5) #constant because we fix the "amplitude of the cycles"

    for epoch in range(n_epoch_cycle):
      optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
      make_epoch(train_loader, cuda, optimizer, net, criterion)
      learning_rate *= np.power(0.9,5/(n_epoch_cycle-1)) #the vitess of decreasing of the learning rate depends of the number of cycle
    learning_rate /= np.power(0.9,5/(n_epoch_cycle-1)) #We cancel the last because the learning rate evolves only n_epoch_cycle-1 times

    #Polyak update of the temp_net
    polyak_update(1/(cycle+2), temp_net, net)

    # Last forward to update BatchNorm layers on temp_net
    make_epoch_no_back(train_loader, cuda, temp_net)
    
    original_acc.append(accuracy(temp_net, original_val, cuda=cuda))
    mixed_same_acc.append(accuracy(temp_net, mixed_same_val, cuda=cuda))
    mixed_rand_acc.append(accuracy(temp_net, mixed_rand_val, cuda=cuda))

  print('Finished Training')
  return original_acc, mixed_same_acc, mixed_rand_acc





def make_training(net, temp_net, n_epoch_first_train, n_cycle, n_epoch_cycle, batch_size = 16, workers = 0, criterion = nn.CrossEntropyLoss(), test_acc_period = 5, _print = True, initial_lr = 1e-4) :
  # Creation of the datasets
  original_dataset = ImageNet9("../data/original")
  original_train = original_dataset.make_loaders(batch_size=batch_size, workers=workers, shuffle_val=True, test = False)
  original_val = original_dataset.make_loaders(batch_size=batch_size, workers=workers)

  mixed_same_dataset = ImageNet9("../data/mixed_same")
  mixed_same_val = mixed_same_dataset.make_loaders(batch_size=batch_size, workers=workers)

  mixed_rand_dataset = ImageNet9("../data/mixed_rand")
  mixed_rand_val = mixed_rand_dataset.make_loaders(batch_size=batch_size, workers=workers)

  use_cuda = True
  if use_cuda and torch.cuda.is_available():
      print("using cuda")
      net.cuda()
      temp_net.cuda()

  # Lauching the training
  original_acc, mixed_same_acc, mixed_rand_acc =  train(net = net, 
                                                        temp_net = temp_net,
                                                        train_loader = original_train,
                                                        original_val = original_val,
                                                        mixed_same_val = mixed_same_val,
                                                        mixed_rand_val = mixed_rand_val,
                                                        n_epoch_first_train = n_epoch_first_train,
                                                        n_cycle = n_cycle,
                                                        n_epoch_cycle = n_epoch_cycle,
                                                        test_acc_period = test_acc_period,
                                                        criterion = criterion,
                                                        _print = _print,
                                                        initial_lr = initial_lr
                                                        )
                                                    
 
  # Display of the results
  print("Final original acc : ", accuracy(net, original_val, cuda=use_cuda))
  print("Final mixed_same acc : ", accuracy(net, mixed_same_val, cuda=use_cuda))
  print("Final mixed_rand acc : ", accuracy(net, mixed_rand_val, cuda=use_cuda))

  print()
  print("Accuracy Graph with ", n_epoch_cycle, " epoch per cycle" )
  plt.plot(range(len(original_acc)), original_acc, label = "original")
  plt.plot(range(len(mixed_same_acc)), mixed_same_acc, label = "mixed_same")
  plt.plot(range(len(mixed_rand_acc)), mixed_rand_acc, label = "mixed_rand")
  plt.legend()
  plt.show()

  return original_acc, mixed_same_acc, mixed_rand_acc


def test_on_dataset(variation, net, batch_size = 16, workers = 0) : 
    dataset = ImageNet9("../data/"+variation)
    val_loader = dataset.make_loaders(batch_size=batch_size, workers=workers)
    return accuracy(net, val_loader, cuda=True)