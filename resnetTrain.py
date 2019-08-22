import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import math
import matplotlib.pyplot as plt
from torchvision import models
import matplotlib.image as mpimg



'''resnet'''
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
class_num = 4
channel_in = model.fc.in_features
model.fc = nn.Linear(channel_in, class_num)


'''Load Data'''
def unpickle(filename):
    with open(filename, 'rb') as f:
        dic = pickle.load(f)
        return dic


train_dict = unpickle('train_batch_crop128.pkl')
test_dict = unpickle('test_batch_crop128_new.pkl')
x_train_all = train_dict['data'].reshape(33470, 3, 128, 128)
y_train_all = train_dict['labels']
x_test = test_dict['data'][:5000].reshape(5000, 3, 128, 128)
y_test = test_dict['labels'][:5000]
print(x_test.shape)
print(y_test.shape)
num_paints = train_dict['paint_number']
print(x_train_all.shape)
print(y_train_all.shape)
train_rate = 0.9
train_count = int(8242*train_rate)

# x_train = x_train_all[:train_count]
# y_train = y_train_all[:train_count]
# x_val = x_train_all[train_count:]
# y_val = y_train_all[train_count:]

x_train = x_train_all[:4500]
y_train = y_train_all[:4500]
x_val = x_train_all[4500:5500]
y_val = y_train_all[4500:5500]


'''normalization'''
x_train_normalized = x_train/255
x_val_normalized = x_val/255
x_test_normalized = x_test/255
print(x_test_normalized.shape)

'''mini-batch preparation'''
print('model structure: ', model)
# init optimizer
learning_rate = 1e-3
# L2weight = 1e-1
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
# set loss function
criterion = nn.CrossEntropyLoss()
# prepare for mini-batch stochastic gradient descent
n_iteration = 10
batch_size = 50
n_data = x_train_normalized.shape[0]
n_batch = int(np.ceil(n_data/batch_size))
print('batch num:', n_batch)

# convert X_train and X_val to tensor and flatten them
# X_train_tensor = Tensor(X_train_normalized).reshape(n_train_data,-1)
# X_val_tensor = Tensor(X_val_normalized).reshape(1000,-1)
X_train_tensor = torch.Tensor(x_train_normalized)
X_val_tensor = torch.Tensor(x_val_normalized)

# convert training label to tensor and to type long
y_train_tensor = torch.Tensor(y_train).long()
y_val_tensor = torch.Tensor(y_val).long()

print('X train tensor shape:', X_train_tensor.shape)


'''training'''

def get_correct_and_accuracy(y_pred, y):
    # y_pred is the nxC prediction scores
    # give the number of correct and the accuracy
    n = y.shape[0]
    # find the prediction class label
    _ ,pred_class = y_pred.max(dim=1)
    correct = (pred_class == y).sum().item()
    return correct ,correct/n

## start
train_loss_list = np.zeros(n_iteration)
train_accu_list = np.zeros(n_iteration)
val_loss_list = np.zeros(n_iteration)
val_accu_list = np.zeros(n_iteration)

for i in range(n_iteration):
    # first get a minibatch of data

    total_train_loss = 0
    total_train_accuracy = 0
    print(i)
    for j in range(n_batch):
        print(j)
        batch_start_index = j * batch_size
        # get data batch from the normalized data
        X_batch = X_train_tensor[batch_start_index:batch_start_index + batch_size]
        # get ground truth label y
        y_batch = y_train_tensor[batch_start_index:batch_start_index + batch_size]
        print(X_batch.shape)
        y_pred = model.forward(X_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()  # 即将梯度初始化为零
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        accu = get_correct_and_accuracy(y_pred, y_batch)[1]
        total_train_accuracy += accu

    ave_train_loss = total_train_loss / n_batch
    train_accu = total_train_accuracy / n_batch

    y_val_pred = model.forward(X_val_tensor)
    val_loss = criterion(y_val_pred, y_val_tensor)  # why y_val_tensor not y_val?
    val_accu = get_correct_and_accuracy(y_val_pred, y_val_tensor)[1]
    print("Iter %d ,Train loss: %.3f, Train acc: %.3f, Val loss: %.3f, Val acc: %.3f"
          % (i, ave_train_loss, train_accu, val_loss, val_accu))
    ## add to the logs so that we can use them later for plotting
    train_loss_list[i] = ave_train_loss
    train_accu_list[i] = train_accu
    val_loss_list[i] = val_loss
    val_accu_list[i] = val_accu

print(train_loss_list)
print(val_loss_list)

x_axis = np.arange(n_iteration)
plt.plot(x_axis, train_loss_list, label='train loss')
plt.plot(x_axis, val_loss_list, label='val loss')
plt.legend()
plt.show()

## plot training accuracy versus validation accuracy
plt.plot(x_axis, train_accu_list, label='train acc')
plt.plot(x_axis, val_accu_list, label='val acc')
plt.legend()
plt.show()

X_test_tensor = torch.Tensor(x_test_normalized)
y_test_tensor = torch.Tensor(y_test).long()
y_test_pred = model.forward(X_test_tensor)
test_accu = get_correct_and_accuracy(y_test_pred,y_test_tensor)[1]
print("Test accuracy: " + str(test_accu) )