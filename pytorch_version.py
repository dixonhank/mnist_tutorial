import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.autograd import Variable
import time

BATCH_SIZE = 128
NUM_EPOCHS = 10

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 3)

        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(40, 10)#（in_features, out_features）

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) 
        # x: 64*1*28*28
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*10*12*12  feature map =[(28-4)/2]^2=12*12
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*20*4*4
        x = F.relu(self.mp(self.conv3(x)))

        x = x.view(in_size, -1) # flatten the tensor 相当于resharp
        # print(x.size())
        # x: 64*320
        x = self.fc(x)
        # x:64*10
        # print(x.size())
        return F.log_softmax(x)  #64*10
    
model = SimpleNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):#batch_idx是enumerate（）函数自带的索引，从0开始
        # data.size():[64, 1, 28, 28]
        # target.size():[64]

        output = model(data)
        #output:64*10


        loss = F.nll_loss(output, target)

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        optimizer.zero_grad()   # 所有参数的梯度清零
        loss.backward()         #即反向传播求梯度
        optimizer.step()        #调用optimizer进行梯度下降更新参数

def test_train():
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        train_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        #print(pred)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    train_loss /= len(train_loader.dataset)
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    
def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        #print(pred)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, 10):
    train(epoch)
    test_train()
    test()
