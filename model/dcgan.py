from model.generator import Generator_factory
from model.discriminator import Discriminator_factory
from torch import nn
import numpy as np
import torch
from torch.optim import *
from torch.utils.data import DataLoader
from skimage import color
from utils import generate_img
import matplotlib.pyplot as plt

def display_training_progress(generator, dataset, epoch):
        test_l, test_ab = dataset[-1]
        test_l = test_l.detach()
        test_ab = test_ab.detach()
        generated_ab = generator.inference(test_l).detach().cpu().numpy()[0]
        test_l = test_l.numpy()
        test_ab = test_ab.numpy()
        # grey img
        grey_img = generate_img(L=test_l, ab=None)
        # AI generated img
        generated_img = generate_img(L=test_l, ab=generated_ab)
        # ground truth img
        test_img = generate_img(L=test_l, ab=test_ab)
        print(np.average(abs(test_ab-generated_ab)))
        # plot
        plt.figure(figsize=(30,30))
        plt.subplot(4,4,1)
        plt.title('B&W')
        plt.imshow(grey_img)
        plt.subplot(4,4,2)
        plt.title('generated')
        plt.imshow(generated_img)
        plt.subplot(4,4,3)
        plt.title('Colored')
        plt.imshow(test_img)
        plt.show()
        # plt.savefig(f"epoch {epoch}")

def set_required_grad(model, flag=True):
    for param in model.parameters():
        param.requires_grad = flag     

def check_grad(model):
    for param in model.parameters():
        if param.grad is None:
            print("no grad")    
        else:
            print(param.grad)

class DCGAN:
    def __init__(self, generator_kwargs, discriminator_kwargs=None, iteration=100, batch_size=64, split_ratio=0.2, discriminator_lr=1e-2, generator_lr=1e-4, discriminator_optm=Adam, generator_optm=Adam, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.discriminator_kwargs = discriminator_kwargs
        self.generator_kwargs = generator_kwargs
        self.discriminator_kwargs["device"]=self.device
        self.generator_kwargs["device"]=self.device
        # initial generator and discriminator
        if discriminator_kwargs:
            self.discriminator = Discriminator_factory(discriminator_kwargs).to(self.device)
            # initial discriminator weights
            self.discriminator.apply(self._init_weights_)
            self.discriminator_optm = discriminator_optm(self.discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))
        self.generator = Generator_factory(self.generator_kwargs).to(self.device)
        # initial generator weights
        self.generator.apply(self._init_weights_)
        # setup training configs
        self.iteration = iteration
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.generator_optm = generator_optm(self.generator.parameters(), lr=generator_lr,  betas=(0.5, 0.999))
        self.l1_loss_fn = nn.L1Loss()
        
    
    def _init_weights_(self, model):
        if isinstance(model, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(model.weight, 0.0, 0.02)
        if isinstance(model, nn.BatchNorm2d):
            nn.init.normal_(model.weight, 0.0, 0.02)
            nn.init.constant_(model.bias, 0)
    
    def train(self, dataset):
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        total_D_loss = 0
        for i in range(self.iteration):
            total_D_loss = 0
            total_G_loss = 0
            total_G_l1_loss = 0
            total_real_loss = 0
            total_fake_loss = 0
            j = 0
            for L, ab in dataloader:
                j += 1
                # generator should always keep training mode
                self.generator.train(True)
                # train discriminator
                self.discriminator.train(True)
                L = L.to(self.device)
                ab = ab.to(self.device)
                fake_ab = self.generator(L)
                train_D_loss, train_real_loss, train_fake_loss = self.discriminator.get_loss(L=L, pred_ab=fake_ab, test_ab=ab)
                ## temporarily freeze the generator model when backward
                set_required_grad(self.generator, False)
                # for bebuging
                self.discriminator.zero_grad()
                train_D_loss.backward(retain_graph=True)
                # for bebuging whether generator model was freezed during backward
                # check_grad(self.generator)
                self.discriminator_optm.step()
                # train generator
                self.discriminator.train(False)
                set_required_grad(self.generator, True)
                D_pred = self.discriminator(L=L, ab=fake_ab)
                train_G_loss, train_G_l1_loss = self.generator.get_loss(fake=fake_ab, real=ab, D_pred=D_pred)
                self.generator_optm.zero_grad()
                train_G_loss.backward()
                # for bebuging whether generator model was unfreezed during backward
                self.generator_optm.step()
                # record loss
                total_D_loss += train_D_loss.item()
                total_G_loss += train_G_loss.item()
                total_G_l1_loss += train_G_l1_loss.item()
                total_real_loss += train_real_loss.item()
                total_fake_loss += train_fake_loss.item()
            print(f"epoch {i}")
            print(f"D_Loss:{total_D_loss/j}, G_Loss:{total_G_loss/j}, train_G_l1_loss:{total_G_l1_loss/j}, train_real_loss:{total_real_loss/j}, train_fake_loss:{total_fake_loss/j}")
            if i % 10 == 0:
                display_training_progress(self, dataset, i)
        
    # input image should be numpy
    @torch.no_grad()
    def inference(self, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(dim=0)
        else:
            image = image
        self.generator.train(False)
        image = image.to(self.device)
        return self.generator(image).cpu()