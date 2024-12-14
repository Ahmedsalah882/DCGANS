
# In this porject, we will do:
# * Build a Generative Adversarial Network (GAN) in PyTorch.
# * Load pre-trained model parameters in PyTorch.
# * Train and test your GAN in PyTorch.
# 
# 
# * Part 1. Importing libraries and downloading the dataset and pre-trained weights
# 
# * Part 2. Build your own GAN 
#   * 2.1 Build the generator 
#   * 2.2 Build the discriminator 
#   * 2.3 Load the pretrained weights 
# 
# * Part 3. Retrain the pre-trained GAN on a new dataset 
#   * 3.1 Preparing the dataset and dataloader
#   * 3.2 Implement the training step for the discriminator 
#   * 3.3 Implement the training step for the generator 
#   * 3.4 Train and evaluate your GAN 
# 
# 
# Dataset
# [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 
# (CelebFaces Attributes Dataset) is a large-scale facial attribute dataset with more than
# 200K celebrity images, each with 40 annotated attributes. 
# The images in this dataset cover significant pose variations and background clutter. 
# In this project, we use the CelebA dataset for pre-training our GAN model.
# 
# [AnimeFace](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) 
# is a dataset consisting of 21551 anime faces scraped from an anime website. 
# All images are resized to 64 × 64 for the sake of convenience. 
# In this project, we perform transfer learning and treat the AnimeFace dataset as
# the downstream dataset.

# Part 1. Importing libraries and downloading the dataset and pre-trained weights

# pip install onedrivedownloader
# get_ipython().system('pip install onedrivedownloader')


import argparse
import os
import random

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils

from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
import time


from onedrivedownloader import download

try:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
except:
    pass



# The following code will switch to CUDA device automatically to accelerate your code
# if GPU is available in your computing environment.
 



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# If you encounter issues with CUDA device, e.g., "RuntimeError: CUDA Out of memory error",
# try to switch the device to CPU by using the following commented code:

# device = torch.device('cpu')

print('Device:', device)


# Finally, we use the [onedrivedownloader](https://pypi.org/project/onedrivedownloader/) package for downloading the data and pre-trained weights.



os.system("pip install onedrivedownloader")

link1 = 'https://unioulu-my.sharepoint.com/:u:/g/personal/jukmaatt_univ_yo_oulu_fi/EXSonItiHilPoo2WequIRCIBr-RdQDTH2xWvmpjbdGisxQ?e=4QbKCv'

link2 = 'https://unioulu-my.sharepoint.com/:u:/g/personal/jukmaatt_univ_yo_oulu_fi/EUvPUbTJW4NNiyc5Nmdf_C0ByyC6eAPf7BdRW_lQE-WDQw?e=lTrBx0'

if not os.path.exists('./data_hw4/anime'):
    print('Downloading the AnimeFace dataset')
    download(link1, filename='./anime.zip', unzip=True, unzip_path='./data_hw4/anime')

if not os.path.exists('./pretrained'):
    print('Downloading pre-trained weights')
    download(link2, filename='./gan_pretrained.zip', unzip=True, unzip_path='./pretrained')

# Part 2. Build our own GAN
# 2.1 Build the generator
# The generator takes a batch of random noise vectors as input and generates 
# an image for each noise vector.
# - The batch of random noise vectors has a shape of `(B, 100)`, where `B` is the batch size
#  and `100` is the length of a single noise vector. 
#  We convert the shape to `(B, 100, 1, 1)`, i.e., `(batch_size, channels, height, width)`,
# to make the input batch suitable for convolutional operations.
# - our generator consists of four convolutional layers,
# expanding the resolution of the feature maps layer by layer and finally obtaining 
# the generated images as outputs with the shape of `(B, 3, 32, 32)`,
#  i.e., `B` three-channel (RGB) images with resolution of 32 × 32 pixels.



class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Define each layer of the Generator 

        # Shapes of each layer: inputs -> outputs, (B, C, H, W)

        # conv1: (B, 100, 1, 1) -> (B, 128, 4, 4)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=128,
                               kernel_size=4, stride=1,padding=0, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
            
        )

        # conv2: (B, 128, 4, 4) -> (B, 64, 8, 8)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                               kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
            
        )

        # conv3: (B, 64, 8, 8) -> (B, 32, 16, 16)
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
            
        )

        # conv4: (B, 32, 16, 16) -> (B, 3, 32, 32)
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3,
                               kernel_size=4, stride=2,padding=1,bias=False),
            nn.Tanh()
            
        )

    def forward(self, x):
        # Finish the forward-pass by using each layer defined above.

        # You can also check the shape of intermediate outputs with the following code.
        # print(x.shape)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        

        return x




# 2.2 Build the discriminator 
# The discriminator takes a batch of images as input and classifies each of them as real or fake.
# - The batch of RGB images has a shape of `(B, 3, 32, 32)`, i.e., `(batch_size, channels, height, width)`.
# - The classification outputs have a shape of `(B, 1, 1, 1)`, which we flatten to a vector with a shape `(B, )` for computing the loss.


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Define each layer of the Discriminator

        # Shapes of each layer: inputs -> outputs, (B, C, H, W)

        # conv1: (B, 3, 32, 32) -> (B, 32, 16, 16)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4,
                      stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2)
            
        )

        # conv2: (B, 32, 16, 16) -> (B, 64, 8, 8)
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                          stride=2,padding=1,bias=False),
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(negative_slope=0.2)
                
        )

        # conv3: (B, 64, 8, 8) -> (B, 128, 4, 4)
        self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                          stride=2,padding=1,bias=False),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU()
        )

        # conv3: (B, 128, 4, 4) -> (B, 1, 1, 1)
        self.conv4 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4,
                          stride=1, padding=0, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        # Finish the forward-pass by using each layer defined above. 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # In the end, we convert the tensor to a vector to make the loss computation friendly
        # flatten: (B, 1, 1, 1) -> (B, )
        x = x.flatten()
        return x



def load_pretrained_weights(model_G, model_D, device, is_debug=False):
    weights_G_path = 'pretrained/weights_G.pth'
    weights_D_path = 'pretrained/weights_D.pth'

    # Complete the code to load pretrained weights from local files. 

    # we call `torch.load()` to load the pretrained weights and pass `weights_G_path` 
    #and `weights_D_path` individually
    #       using the correct `device` to the `map_location`.
    weights_G = torch.load(weights_G_path)
    weights_D = torch.load(weights_D_path)

    # we call `load_state_dict()` and pass `weights_G` and `weights_D` individually.
    model_G.load_state_dict(weights_G)
    model_D.load_state_dict(weights_D)


    if is_debug:
        print('The type of weights_D:\n', type(weights_D), '\n')
        print('The keys in weights_D:\n', list(weights_D.keys()), '\n')
        print('The shape of conv1.0 in weights_D:\n', weights_D['conv1.0.weight'].shape)



# Next, we will make a visual comparison between the outputs of the randomly initialized 
# and pre-trained generators.


fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Create the instances of Generator and Discriminator
model_G = Generator().to(device)
model_D = Discriminator().to(device)

# Create a set of fixed noise vectors for visualization
fixed_noise = torch.randn((36, 100, 1, 1), device=device)

# Generate images with the random initialized Generator and display them
generated_images = model_G(fixed_noise)
generated_images = utils.make_grid(generated_images.detach().cpu(), padding=2,
                                   normalize=True, nrow=6)

ax[0].axis('off')
ax[0].set_title('Generated images (random initialized)')
ax[0].imshow(np.transpose(generated_images, (1, 2, 0)))

# Load the pre-trained weights for the Generator and Discriminator
load_pretrained_weights(model_G, model_D, device)

# Generate images with the pre-trained Generator and display them
generated_images_pretrained = model_G(fixed_noise)
generated_images_pretrained = utils.make_grid(generated_images_pretrained.detach().cpu(),
                                              padding=2, normalize=True, nrow=6)

ax[1].axis('off')
ax[1].set_title('Generated images (pre-tained)')
ax[1].imshow(np.transpose(generated_images_pretrained, (1, 2, 0)))


# If the pre-trained weights have been loaded correctly, you should see faces 
# in the generated images on the right instead of random noise 
# (as in the case of random initialization on the left).


# Part 3. Retrain the pre-trained GAN on a new dataset

# 3.1 Preparing the dataset and dataloader

# First, let's define some hyperparameters for the data preprocessing and training.


# Image size
image_size = 32

# Batch size for training
batch_size = 128
num_workers = 1

# Learning rate for optimizers
lr = 0.0002

# Number of training epochs
num_epochs = 30


# Then, let's create the dataset and dataloader to load the AnimeFace dataset for training.


# We can use the ImageFolder class due to the structure of the AnimeFace dataset
# Create the dataset
dataset = torchvision.datasets.ImageFolder(
    root='./data_hw4',
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)
print("---"*45 + "\n")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(5, 5))
plt.axis('off')
plt.title('The training dataset')
plt.imshow(np.transpose(utils.make_grid(real_batch[0].to(device)[:36], padding=2, normalize=True, nrow=6).cpu(),(1,2,0)))
print("---"*45 + "\n")


# Finally, we integrate all the model definition and initialization steps into one function `init_model_and_optimizer()`.

def init_model_and_optimizer():

    # Create the instances of Generator and Discriminator
    model_G = Generator().to(device)
    model_D = Discriminator().to(device)

    # Load the pre-trained weights for model_G and model_D
    load_pretrained_weights(model_G, model_D, device)

    # Setup Adam optimizers for both model_G and model_D
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Initialize the loss function for training
    BCE_loss = nn.BCELoss()

    return model_G, model_D, optimizer_G, optimizer_D, BCE_loss    
    

model_G, model_D, optimizer_G, optimizer_D, BCE_loss = init_model_and_optimizer()


# 3.2 Implement the training step for the discriminator 
# The training data for the discriminator comes from two sources:
# - Real images, such as real pictures of faces: The discriminator uses these instances
# as positive examples during training.
# - Fake images generated by the generator: The discriminator uses these instances 
# as negative examples during training.
# 
#   During the training step:
# - The discriminator classifies both **real images and fake images** from the generator.
# - The discriminator loss penalizes the discriminator for misclassifying a real instance 
# as fake or a fake instance as real.
# - The discriminator updates its weights through backpropagation from the 
# discriminator loss through the discriminator network.



def training_step_D(
    real_images,
    model_G: nn.Module,
    model_D: nn.Module,
    optimizer_D: torch.optim.Optimizer,
    BCE_loss: nn.BCELoss,
    is_debug=False,
):
    """Method of the training step for Discriminator.

    Args:
        real_images: a batch of real image data from the training dataset
        model_G: the generator model
        model_D: the discriminator model
        optimizer_D: optimizer of the Discriminator
        BCE_loss: binary cross entropy loss function for loss computation

    Returns:
        loss_D: the discriminator loss

    """

    # Reset the gradients of all parameters in discriminator
    model_D.zero_grad()

    batch_size = real_images.shape[0]

    # Prepare the real images and their labels
    real_images = real_images.to(device)
    real_labels = torch.ones((batch_size, ), device=device)

    # Prepare the fake images and their labels using the generator and the random `noise`
    # (1) pass the `noise` to the `model_G` to get `fake_images`.
    # (2) create the labels for the fake images with `torch.zeros()`.
   
    noise = torch.randn((batch_size, 100, 1, 1), device=device)
    fake_images = model_G(noise)
    fake_labels = torch.zeros((batch_size,),device=device)

    # we call `model_D()` and `BCE_loss` to calculate the losses for real and fake images
    # (1) pass `real_images` into `model_D()` to get the `real_outputs`.
    # (2) calculate the `loss_D_real` with `real_outputs` and `real_labels `using` BCE_loss.
    # (3) calculate the `loss_D_fake` following a similar way as (1) & (2).
    
    real_outputs = model_D(real_images) 
    loss_D_real =  BCE_loss(real_outputs, real_labels)

    fake_outputs = model_D(fake_images)
    loss_D_fake =  BCE_loss(fake_outputs, fake_labels)


    # Sum the loss of both real images and fake images to get the total discriminator loss.
    loss_D = loss_D_fake + loss_D_real

    # Compute the gradients
    loss_D.backward()

    # Update the parameters of `model_D`
    optimizer_D.step()

    if is_debug:
        print('Shape of real outputs:\n', real_outputs.shape, '\n')
        print('Shape and samples of real labels:\n', real_labels.shape, ' ', real_labels[:5], '\n')

        print('Shape of fake outputs:\n', fake_outputs.shape, '\n')
        print('Shape and samples of fake labels:\n', fake_labels.shape, ' ', fake_labels[:5], '\n')

    return loss_D


# 3.3 Implement the training step for the generator 

# The training process for the generator requires tighter integration 
# between the generator and the discriminator compared with the training step for 
# the discriminator. The portion of the GAN that trains the generator includes:
# - Random noise as input
# - The generator network for transforming the random input into a data instance
# - The discriminator network for classifying the generated data and its output
# - Generator loss for penalizing the generator for failing to fool the discriminator


def training_step_G(
    model_G: nn.Module,
    model_D: nn.Module,
    optimizer_G: torch.optim.Optimizer,
    BCE_loss: nn.BCELoss,
    is_debug=False,
):
    """Method of the training step for Generator.

    Args:
        model_G: the generator model
        model_D: the discriminator model
        optimizer_G: optimizer for the generator
        BCE_loss: binary cross entropy loss function for loss computation

    Returns:
        loss_G: the generator loss

    """

    # Reset the gradients of all parameters in `model_G`
    model_G.zero_grad()

    # Generate fake images from `model_G` with random noises
    noise = torch.randn((batch_size, 100, 1, 1), device=device)
    fake_images = model_G(noise)


    # Prepare labels for fake_images 

    labels = torch.ones((batch_size, ), device=device)


    # we call `model_D()` and `BCE_loss` to calculate the loss of Generator 
    # (1) pass `fake_images` into `model_D()` to get the `outputs`.
    # (2) calculate the `loss_G` with `outputs` and `labels` using `BCE_loss()`.
    outputs = model_D(fake_images)
    loss_G = BCE_loss(outputs, labels)


    # Compute the gradients
    loss_G.backward()

    # Update the parameters of `model_G`
    optimizer_G.step()

    if is_debug:
        print('Shape of outputs:\n', outputs.shape, '\n')
        print('Shape of labels:\n', labels.shape, '\n')

    return loss_G


# 3.4 Train and evaluate your GAN  

# Create the model, optimizer, and loss funtions for training

model_G, model_D, optimizer_G, optimizer_D, BCE_loss = init_model_and_optimizer()


# Lists and variables to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

torch.random.seed()
start_time = time.time()


# Training Loop

print("Starting the training loop...")

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (real_images, _) in enumerate(dataloader, 0):

        # call the `training_step_D()` and `training_step_G()` 
        # and collect the loss values `loss_D` and `loss_G` 
        # pass variables required by these two functions, including:
        # `real_images`, `model_G`, `model_D`, `optimizer_D`, `optimizer_G`, and `BCE_loss`.
        loss_D = training_step_D(real_images, model_G, model_D, optimizer_D, BCE_loss) 
        loss_G =training_step_G(model_G, model_D, optimizer_G, BCE_loss)

        # Output training stats
        if i % 50 == 0:
            print('[Epoch][Iter][{}/{}][{}/{}] Loss_D: {:.4f}, Loss_G: {:.4f}, Time: {:.2f} s'.format(
                epoch, num_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), time.time() - start_time))
            start_time = time.time()

        # Save losses for plotting later
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 50 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = model_G(fixed_noise).detach().cpu()
            img_list.append(utils.make_grid(fake, padding=2, normalize=True, nrow=6))
        iters += 1

print("Training finished!")
print("---"*45 + "\n")


# Let's plot the generator and discriminator losses during training our GAN.

plt.figure()
plt.title("Generator and discriminator losses during training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
print("---"*45 + "\n")


# Let's inspect how the generated images look like after the training of our GAN has finished.

fig = plt.figure(figsize=(5, 5))
plt.axis("off")

ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]

