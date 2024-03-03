import torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# 
def Generator_factory(kwargs):
    if kwargs["model"] == "UNet":
        return UNetGenerator(**kwargs)
    else:
        return None

# abstract class of generator  
class Generator(nn.Module):
    def __init__(self, num_filters, num_colours, num_in_channels, kernel,pixel_loss_coef=1, model="Generator", Loss_fn=nn.L1Loss, device="cpu"):
        super(Generator, self).__init__()
        self.num_filters = num_filters
        self.num_colours = num_colours
        self.num_in_channels = num_in_channels
        self.kernel = kernel
        self.name = model
        self.pixel_loss_coef = pixel_loss_coef
        self.device = device
        self.Loss_fn = Loss_fn()
        self.d_loss = nn.BCELoss()
    
    def forward(self, X):
        pass
    
    # y:target image, x:input image, z = G(X): generated image
    # goal: make D(x, z=G(x)) close to 1, and MAE(z, y) close to 0
    # G Loss = D_Loss(BCE(D(G(x)=z), 1))+ MAE(true pics, generated pics)(AKA L1)
    def get_loss(self, fake, real, D_pred=None):
        # Generator loss = MAE(fake, real) + Cross_entropy(D_pred, classifiy as G(x) as fakes)
        pixel_loss = self.Loss_fn(fake, real)
        if D_pred is None:
            G_loss = pixel_loss
            return G_loss
        else:
            # try to make discriminator predict the generated images as real images
            labels = torch.ones_like(D_pred).to(self.device)
            d_loss = self.d_loss(D_pred, labels)
            G_loss = self.pixel_loss_coef*pixel_loss+d_loss
            return G_loss, pixel_loss
        
# block for compressing image
class UNetCompressBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel, padding, stride, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNetCompressBlock, self).__init__()
        model = []
        self.Conv2d = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel, padding=padding, stride=stride)
        model.append(self.Conv2d)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        model.append(self.relu)
        if norm_layer:
            model.append(norm_layer(output_channel))
        if use_dropout:
            model.append(nn.Dropout2d(p=0.2))
        self.model = nn.Sequential(*model)
        
    def forward(self, X):
        return self.model(X)

# block for uncompressing image
class UNetTransposeBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel, padding, stride, output_padding, norm_layer=nn.BatchNorm2d, use_dropout=True):
        super(UNetTransposeBlock, self).__init__()
        model = []
        self.ConvTranspose2d = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=kernel, padding=padding, stride=stride, output_padding=output_padding)
        model.append(self.ConvTranspose2d)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        model.append(self.relu)
        if norm_layer:
            model.append(norm_layer(output_channel))
        if use_dropout:
            model.append(nn.Dropout2d(p=0.2))
        self.model = nn.Sequential(*model)
    
    def forward(self, X):
        return self.model(X)
        
class UNetGenerator(Generator):
    def __init__(self, num_filters, num_colours, num_in_channels, kernel, pixel_loss_coef, model, Loss_fn=nn.L1Loss, device="cpu", use_dropout=False):
        super(UNetGenerator, self).__init__(num_filters=num_filters, num_colours=num_colours, num_in_channels=num_in_channels, kernel=kernel, pixel_loss_coef=pixel_loss_coef, model=model, Loss_fn=Loss_fn, device=device)
        self.kernel = kernel
        self.stride = 2
        self.padding = 1
        self.output_padding = 0
        self.num_filters = num_filters
        self.num_colours = num_colours
        self.num_in_channels = num_in_channels
        # compress block 1, In order to reserve the input image information, dont apply batch normalization in the first layer
        self.unet_compress_block1 = UNetCompressBlock(self.num_in_channels, self.num_filters, self.kernel, self.padding, self.stride, norm_layer=None, use_dropout=False)
        # compress block 2
        self.unet_compress_block2 = UNetCompressBlock(self.num_filters, 2*self.num_filters, self.kernel, self.padding, self.stride)
        # compress block 3
        self.unet_compress_block3 = UNetCompressBlock(2*self.num_filters, 4*self.num_filters, self.kernel, self.padding, self.stride)
        # compress block 4
        self.unet_compress_block4 = UNetCompressBlock(4*self.num_filters, 4*self.num_filters, self.kernel, self.padding, self.stride)
        # compress block 5
        self.unet_compress_block5 = UNetCompressBlock(4*self.num_filters, 4*self.num_filters, self.kernel, self.padding, self.stride, norm_layer=None, use_dropout=False)
        self.downs = [self.unet_compress_block1, self.unet_compress_block2, self.unet_compress_block3, self.unet_compress_block4, self.unet_compress_block5]
        # transpose block 1
        self.unet_transpose_block1 = UNetTransposeBlock(4*self.num_filters, 4*self.num_filters, self.kernel, self.padding, self.stride, self.output_padding)
        # transpose block 1
        self.unet_transpose_block2 = UNetTransposeBlock(8*self.num_filters, 4*self.num_filters, self.kernel, self.padding, self.stride, self.output_padding)
        # transpose block 2
        self.unet_transpose_block3 = UNetTransposeBlock(8*self.num_filters, 2*self.num_filters, self.kernel, self.padding, self.stride, self.output_padding)
        # transpose block 3
        # input channel = output channel of compress block 1 + output channel of transpose block 1
        self.unet_transpose_block4 = UNetTransposeBlock(4*self.num_filters, self.num_filters, self.kernel, self.padding, self.stride, self.output_padding, use_dropout=False)
        self.unet_transpose_output = UNetTransposeBlock(2*self.num_filters, self.num_colours, self.kernel, self.padding, self.stride, self.output_padding, norm_layer=None, use_dropout=False)
        self.ups = [self.unet_transpose_block1,  self.unet_transpose_block2, self.unet_transpose_block3,  self.unet_transpose_block4]
        # final conv2d layer
        # input channel = input channel of compress block 1(orginal picture) + output channel of transpose block 2
        self.tanh = nn.Tanh()
        
    def forward(self, X):
        skips = []
        for down in self.downs:
            X = down(X)
            skips.append(X)
        skips.pop()
        for skip, up in zip(skips[::-1], self.ups):
            X = up(X)
            X = torch.cat([X, skip], dim=1)
        X = self.unet_transpose_output(X)
        return self.tanh(X)
    
    
    
    
if __name__ == "__main__":
    """kwargs = {
        "model":"UNet",
        "num_filters": 4, 
        "num_colours":3, 
        "num_in_channels":1, 
        "kernel":2
    }
    generator = Generator_factory(kwargs)
    image = torch.rand(1,1, 64, 64)
    print(generator(image).shape)"""
      