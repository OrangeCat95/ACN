import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchsummary import summary
#from logger import Logger

# 定义超参数
batch_size = 64        # 批的大小
learning_rate = 1e-2    # 学习率
num_epoches = 1000        # 遍历训练集的次数

# 数据类型转换，转换成numpy类型
#def to_np(x):
#    return x.cpu().data.numpy()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
transform_test = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.CIFAR100(
    root='./data', train=True, transform=transform_train, download=True)

test_dataset = datasets.CIFAR100(
    root='./data', train=False, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义 Convolution Network 模型
class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()    # super用法:Cnn继承父类nn.Model的属性，并用父类的方法初始化这些属性
        self.conv1 = nn.Sequential(     #padding=2保证输入输出尺寸相同(参数依次是:输入深度，输出深度，ksize，步长，填充)
            nn.Conv2d(in_dim, 96, 3, stride=1, padding=1),
            nn.Dropout(p=0.2),
            nn.ReLU(True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), stride=2),
            nn.Dropout(p=0.5))

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 192, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), stride=2),
            nn.Dropout(p=0.5))

        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(192, 100, 1, stride=1, padding=0))

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),)
            # nn.Softmax())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x



model = Cnn(3, 100)  # 图片大小是28x28,输入深度是1，最终输出的10类

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
summary(model, (3, 32, 32))
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
    print("Using GPU now!")

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()

weight_p, bias_p = [],[]
for name, p in model.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

optimizer = optim.SGD([
    {'params': weight_p, 'weight_decay':1e-3},
    {'params': bias_p, 'weight_decay':0}
    ], lr=1e-2, momentum=0.9, nesterov=True)


print("Model's state_dict:")
# Print model's state_dict
for param_tensor in model.state_dict():
    print(param_tensor,"\t",model.state_dict()[param_tensor].size())
print("optimizer's state_dict:")
# Print optimizer's state_dict
for var_name in optimizer.state_dict():
    print(var_name,"\t",optimizer.state_dict()[var_name])


eval_best = 0
#logger = Logger('./logs')
# 开始训练
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))      # .format为输出格式，formet括号里的即为左边花括号的输出
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        # cuda
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        """
        # ========================= Log ======================
        step = epoch * len(train_loader) + i
        # (1) Log the scalar values
        info = {'loss': loss.data[0], 'accuracy': accuracy.data[0]}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)

        # (3) Log the images
        info = {'images': to_np(img.view(-1, 28, 28)[:10])}

        for tag, images in info.items():
            logger.image_summary(tag, images, step)
        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
        """
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))
    model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_acca = 0
    for data in test_loader:
        img, label = data
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))


    eval_acca = eval_acc / (len(test_dataset))
    # print("eval_acca={:.4f}".format(eval_acca))
    if eval_acca > eval_best:
        eval_best = eval_acca
        torch.save(model.state_dict(), './cifar10.pth')
        print("Save the best model weights, acc={:.6f}".format(eval_best))
    else:
        print("Eval acc don't raise from {:.4f}, don't save model weight".format(eval_best))
    #     pass

print("The final best eval acca is {:.4f}".format(eval_best))




# 保存模型





