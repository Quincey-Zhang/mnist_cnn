import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt

from model import CNN


DOWNLOAD_MNIST = False  # if need to download data, set True at first time
BATCH_SIZE = 50
EPOCH = 5
LR = 1e-3


""" data """

# read train data
train_data = torchvision.datasets.MNIST(
    root='./data', train=True, download=DOWNLOAD_MNIST, transform=torchvision.transforms.ToTensor())
print()
print("size of train_data.train_data:  {}".format(train_data.train_data.size()))  # train_data.train_data is a Tensor
print("size of train_data.train_labels:  {}".format(train_data.train_labels.size()), '\n')

# plot one example
idx_example = 10
plt.imshow(train_data.train_data[idx_example].numpy(), cmap='Greys')
plt.title('{}'.format(train_data.train_labels[idx_example]))
# plt.show()

# data loader
# combines a dataset and a sampler, and provides an iterable over the given dataset
train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# read test data
test_data = torchvision.datasets.MNIST(root='./data', train=False)
num_test, num_test_over = 2000, 1000
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(
    torch.FloatTensor)[: num_test] / 255.  # unsqueeze because of 1 channel; value in range (0, 1)
test_y = test_data.test_labels[: num_test]
test_over_x = torch.unsqueeze(
    test_data.test_data, dim=1).type(torch.FloatTensor)[-num_test_over:] / 255.  # test data after training
test_over_y = test_data.test_labels[-num_test_over:]


""" train """

cnn = CNN()
print("CNN model structure:\n")
print(cnn, '\n')
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=LR)  # optimizer: Adam
loss_func = nn.CrossEntropyLoss()  # loss function: cross entropy

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):  # x, y: Tensor of input and output
        output = cnn(x)
        loss = loss_func(output, y)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # test at each 50 steps
        if not step % 50:
            test_output = cnn(test_x)
            predict_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((predict_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.3f' % accuracy)


# output model file
torch.save(cnn, 'cnn.pt')
print()
print('finish training', '\n')


""" test """

# load model
print('load cnn model', '\n')
cnn_ = torch.load('cnn.pt', weights_only=False)

# test new data
test_output = cnn_(test_over_x)
predict_y = torch.max(test_output, 1)[1].data.numpy()
print("prediction number:  {}".format(predict_y))
print("real number:  {}".format(test_over_y.numpy()), '\n')
accuracy = float((predict_y == test_over_y.data.numpy()).astype(int).sum()) / float(test_over_y.size(0))
print("accuracy:  {}".format(accuracy), '\n')
