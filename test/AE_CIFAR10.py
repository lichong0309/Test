import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("test")

num_epochs = 50
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])     

trainset = tv.datasets.CIFAR10(root='./data',  train=True, download=True,  transform=transform)
testset  = tv.datasets.CIFAR10(root='./data',  train=False, download=True,  transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Encoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)
        
class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))

class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()
        self.m1 = DecoderModule(256, 128, stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")

    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)

class Autoencoderv3(nn.Module):
    def __init__(self):

        super().__init__()
        # # latent features
        self.n_latent_features = 64

        # resolution
        # mnist, fashion-mnist : 28 -> 14 -> 7
        # cifar : 32 -> 8 -> 4
        # stl : 96 -> 24 -> 6


        pooling_kernel = [4, 2]
        encoder_output_size = 4

        color_channels = 3

        # # neurons int middle layer
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)
    
    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def _bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar
        
    def sampling(self):
        # assume latent features space ~ N(0, 1)
        z = torch.randn(64, self.n_latent_features).to(self.device)
        z = self.fc3(z)
        # decode
        return self.decoder(z)

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        # Bottle-neck
        z, mu, logvar = self._bottleneck(h)
        # decoder
        z = self.fc3(z)
        d = self.decoder(z)
        return d, h


def loss_function(recon_x, x, mu, logvar):
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)        
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

model = Autoencoderv3().to(device)
distance   = nn.MSELoss()
class_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


mse_multp = 0.5
cls_multp = 0.5

test_acc = []    # initialize

model.train()

for epoch in range(num_epochs):
    total_mseloss = 0.0
    total_clsloss = 0.0
    for ind, data in enumerate(dataloader):
        img, labels = data[0].to(device), data[1].to(device)
        output, output_en = model(img)
        loss_mse = distance(output, img)
        loss_cls = class_loss(output_en, labels)
        loss = (mse_multp * loss_mse) + (cls_multp * loss_cls)  # Combine two losses together
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Track this epoch's loss
        total_mseloss += loss_mse.item()
        total_clsloss += loss_cls.item()

    # Check accuracy on test set after each epoch:
    model.eval()   # Turn off dropout in evaluation mode
    acc = 0.0
    total_samples = 0
    for data in testloader:
        # We only care about the 10 dimensional encoder output for classification
        img, labels = data[0].to(device), data[1].to(device)
        _, output_en = model(img)   
        # output_en contains 10 values for each input, apply softmax to calculate class probabilities
        prob = nn.functional.softmax(output_en, dim = 1)
        pred = torch.max(prob, dim=1)[1].detach().cpu().numpy() # Max prob assigned to class 
        acc += (pred == labels.cpu().numpy()).sum()
        total_samples += labels.shape[0]
    model.train()   # Enables dropout back again
    test_acc_temp = acc / total_samples
    print('epoch [{}/{}], loss_mse: {:.4f}  loss_cls: {:.4f}  Acc on test: {:.4f}'.format(epoch+1, num_epochs, total_mseloss / len(dataloader), total_clsloss / len(dataloader), acc / total_samples))
    test_acc.append(test_acc_temp)

x = []
for i in range(len(test_acc)):
    x.append(i)

fig = plt.figure()
plt.xlabel('epoch')
plt.ylabel("accuracy")

plt.plot(x, test_acc, 's-', color='r')
plt.savefig('./Figure/CIFAR10.png')
plt.show()

