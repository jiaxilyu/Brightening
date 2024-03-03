import torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def Discriminator_factory(kwargs):
    if kwargs["model"] == "ImageDiscriminator":
        return ImageDiscriminator(**kwargs)
    elif kwargs["model"] == "PixDiscriminator":
        return PixDiscriminator(**kwargs)
    else:
        return None
    
class cnn_block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=4, stride=2, padding=1, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(cnn_block, self).__init__()
        use_bias = False
        if norm_layer:
            use_bias = norm_layer == nn.InstanceNorm2d
            model = [norm_layer(output_channel, momentum=0.1,eps=1e-5)]
        else:
            model = []
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.relu = nn.LeakyReLU(0.2, True)
        model = [self.conv]+model+[self.relu]
        if use_dropout:
           model.append(nn.Dropout2d(p=0.2))
        self.model = nn.Sequential(*model)
    
    def forward(self, X):
        return self.model(X)
    
class Discriminator(nn.Module):
    def __init__(self, model, image_channel, device="cpu", Loss_fn=nn.BCEWithLogitsLoss):
        super(Discriminator, self).__init__()
        self.name = model
        self.in_channels = image_channel
        self.device = device
        self.loss_fn = Loss_fn()
    
    # y: true image, x:input image, z = G(X): generated image
    # goal: make D(G(x)=z) close to 0, and D(y) close to 1
    # goal = true_loss(BCE(D(y), 1)) + fake_loss(BCE(z=G(x), 0))
    def get_loss(self, L, pred_ab, test_ab):
        # true label loss
        y_true_pred = self.forward(L=L, ab=test_ab)
        true_labels = torch.ones_like(y_true_pred).to(self.device)
        true_D_loss = self.loss_fn(y_true_pred, true_labels)
        # fake label loss
        y_fake_pred = self.forward(L=L, ab=pred_ab)
        fake_labels = torch.zeros_like(y_fake_pred).to(self.device)
        fake_D_loss = self.loss_fn(y_fake_pred, fake_labels)
        # get sum loss
        D_loss = (true_D_loss + fake_D_loss)/2
        return D_loss, true_D_loss, fake_D_loss
        
# Predict the probability of the input image being real  
class ImageDiscriminator(Discriminator):
    def __init__(self, model, image_channel, device="cpu", Loss_fn=nn.BCELoss):
        super(ImageDiscriminator, self).__init__(model=model, image_channel=image_channel, device=device, Loss_fn=Loss_fn)
        # channel = 1 only black and white, channel = 3 colorful(RGB)
        # nn.Conv2d(channel, filters, kernel_x, kernel_y)
        self.conv1 = cnn_block(self.in_channels, 64, kernel_size=4, stride=2, padding=1, norm_layer=None)
        self.conv2 = cnn_block(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = cnn_block(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = cnn_block(256, 512, kernel_size=4, stride=2, padding=1, norm_layer=None)
        self.output = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Flatten(),
                                   nn.Linear(512, 1),
                                   nn.Sigmoid())
    
    # X is the L, 
    def forward(self, L, ab):
        X = torch.cat([L, ab], dim=1)
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        return self.output(X)
    
    def get_fc_input_size(self):
        return self.conv2(self.conv1(torch.autograd.Variable(torch.zeros(*self.image_shape)))).view(1, -1).size(1)

# Predict the probability of every pixel in the input image being real
class PixDiscriminator(Discriminator):
    def __init__(self, model, image_channel, device="cpu", Loss_fn=nn.BCELoss):
        super(PixDiscriminator, self).__init__(model=model, image_channel=image_channel, device=device, Loss_fn=Loss_fn)
        self.conv1 = cnn_block(self.in_channels, 64, kernel_size=4, stride=2, norm_layer=None, use_dropout=False)
        self.conv2 = cnn_block(64, 128, kernel_size=4, stride=2)
        self.conv3 = cnn_block(128, 256, kernel_size=4, stride=2)
        self.conv4 = cnn_block(256, 512, kernel_size=4, stride=1)
        self.output = cnn_block(512, 1, kernel_size=4, stride=1, norm_layer=None, use_dropout=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, L, ab):
        X = torch.cat([L, ab], dim=1)
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.output(X)
        return self.sigmoid(X)
    
    
# if __name__ == "__main__":
#     image_shape = (3, 64, 64)
#     discriminator = Discriminator(image_shape)
#     l = torch.rand(1,1, 64, 64)
#     ab = torch.rand(1,2, 64, 64)
#     label = torch.ones(1,1)
#     loss_fn = nn.BCEWithLogitsLoss()
#     pred = discriminator(l, ab)
#     loss = loss_fn(pred, label)
#     print(loss)
#     discriminator.requires_grad_(False)
#     loss.backward(retain_graph=True)
#     for model in discriminator.parameters():
#         print(model.grad)
#     discriminator.requires_grad_(True)
#     loss.backward()
#     for model in discriminator.parameters():
#         print(model.grad)