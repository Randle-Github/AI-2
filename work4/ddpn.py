# Commented out IPython magic to ensure Python compatibility.
import os
import torch
import torch.nn as nn
import argparse
import numpy as np

#%cd /content/drive/MyDrive/Fifth year/ClearBox/Diffusion_model_training/Cifar100_model
# %cd /content/drive/MyDrive/Diffusion_model_training/Cifar100_model
from model import UNET
#from model_test import UNet_conditional
from torchvision.datasets import CIFAR100, MNIST,CIFAR10
from utils import sin_time_embeding, beta_schedule
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import ToPILImage
import torchvision
from torchsummary import summary
from tqdm import tqdm
import sys
from PIL import Image



class basics:
  def __init__(self, args, number_noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, image_size = 64, device = "cuda"):
    self.number_noise_steps = args.number_noise_steps
    self.beta_start = args.beta_start
    self.beta_end = args.beta_end
    self.image_size = args.image_size
    self.device = args.device
    self.args = args

    schedule = beta_schedule(self.beta_start, self.beta_end, self.number_noise_steps)

    if args.noise_schedule == "linear":
      self.beta = schedule.linear()
    elif args.noise_schedule == "quadratic":
      self.beta = schedule.quadratic()
    elif args.noise_schedule == "sigmoid":
      self.beta = schedule.sigmoid() 
    elif args.noise_schedule == "cosine":
      self.beta = schedule.cosine()
      
    self.beta = self.beta.to(args.device)
    self.alpha = 1 - self.beta
    self.big_alpha = torch.cumprod(self.alpha, dim = 0)
  
  
  def produce_noise(self, x, time_position):
    part1 = torch.sqrt(self.big_alpha[time_position])[:, None, None, None]
    part2 = torch.sqrt(1 - self.big_alpha[time_position])[:, None, None, None]
    noise = torch.randn_like(x)
    return part1 * x + part2 * noise, noise

  def sampling_image(self, model, batch_size, label, classifier_scale = 3): #Labels has to have batch size
    print("Start Sampling")
    model.eval()
    x_noise = torch.randn(batch_size, 3, self.image_size, self.image_size)
    with torch.no_grad(): 
      for i in tqdm(reversed(range(1, self.number_noise_steps))):
        t = (torch.ones(batch_size) * i).long()
        t = t.to(self.args.device)
        if i == 0:
          z = torch.zeros(x_noise.size())
        else:
          z = torch.randn_like(x_noise)

        alpha_buffer = self.alpha[t][:, None, None, None]
        big_alpha_buffer = self.big_alpha[t][:, None, None, None]
        beta_buffer = self.beta[t][:, None, None, None]
        
        t = t.unsqueeze(-1).type(torch.float)
        sinusoidal_time_embeding = sin_time_embeding(t, device = self.args.device).to(self.args.device)

        x_noise = x_noise.to(self.args.device)
        z = z.to(self.args.device)

        pred_classified_noise = model(x_noise, sinusoidal_time_embeding, label)
        pred_noise = pred_classified_noise

        if classifier_scale > 0:  #The classifier scale is what defines the intensity of the interpolation towards the classified predicited noise
          pred_unclassified_noise = model(x_noise, sinusoidal_time_embeding, None)
          pred_interpolated_noise = torch.lerp(pred_unclassified_noise, pred_classified_noise,classifier_scale)
          pred_noise = pred_interpolated_noise

        part2 = ((1 - alpha_buffer)/(torch.sqrt(1 - big_alpha_buffer))) * pred_noise
        xtm = ((1/torch.sqrt(alpha_buffer)) * (x_noise - part2)) + torch.sqrt(beta_buffer) * z
        x_noise = xtm
      x_noise = (x_noise.clamp(-1, 1) + 1) / 2
      x_noise = (x_noise * 255).type(torch.uint8)
    model.train()
    return x_noise

def train(args, model,dataloader, optmizer, loss, model_checkpoint = None):#Need to take it out of the basics object and create args 
  basic_obj = basics(args, args.number_noise_steps, args.beta_start, args.beta_end, args.image_size, args.device)
  
  if args.use_checkpoints == "True" and model_checkpoint != None:  #Load the checkpoints of the model
    print("Using checkpoint")
    model.load_state_dict(model_checkpoint['model_state_dict'])
    optmizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
    epoch = model_checkpoint["epoch"]
  else:
    epoch = 0

  while epoch < args.number_epochs:
    print("Epoch: ", epoch)
    list_losses = []
    for i, data in tqdm(enumerate(dataloader)):   #Iterating over the images from the dataloader
      optmizer.zero_grad()             #Setting gradient to zero after each iteration
      
      label = data[1].to(args.device)
      x0 = data[0].to(args.device)

      t = torch.randint(1, args.number_noise_steps, (args.batch_size, )).to(args.device)  #Getting a vector of time values the size of the bactch
      
      xt_rand, normal_distribution = basic_obj.produce_noise(x0, t)   #Generaring the noisy image at the specified time stamps from vector "t"
      xt_rand = xt_rand.to(args.device)
      normal_distribution = normal_distribution.to(args.device)

      t = t.unsqueeze(-1).type(torch.float)
      sinusoidal_time_embeding = sin_time_embeding(t, device = args.device).to(args.device) #This needs to be done because the UNET only accepts the time tensor when it is transformed

      if torch.rand(1) < 0.1:
        label = None

      x_pred = model(xt_rand, sinusoidal_time_embeding, label).to(args.device)    #Predicted images from the UNET by inputing the image and the time without the sinusoidal embeding
      
      Lsimple = loss(normal_distribution, x_pred).to(args.device)
      list_losses.append(Lsimple.item())
      Lsimple.backward()
      optmizer.step()

    labels_to_predict = torch.tensor(4).to(args.device)
    image_sample = basic_obj.sampling_image(model, 1, labels_to_predict)
    image_sample1 = torch.squeeze(image_sample)

    trsmr = ToPILImage()
    img_pil1 = trsmr(image_sample1)
    display(img_pil1)
    epoch += 1
   
    
    #Saving Checkpoint
      
    EPOCH = epoch
    PATH = args.checkpoint_directory + "/" + args.noise_schedule + "_" + "DiffusionModel.pt"      
    torch.save({
        'epoch': EPOCH,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optmizer.state_dict(),
        'LRSeg': args.learning_rate,
        }, PATH)
    print("checkpoint saved")
  

    
    print("The average loss was: ", np.mean(list_losses))

def main(params):
  parser = argparse.ArgumentParser(description='Diffusion model')

  parser.add_argument('--device', type=str, default="cuda", help='Device to run the code on')
  parser.add_argument('--use_checkpoints', type=str, default="False", help='Use checkpoints')
  parser.add_argument('--emb_dimension', type=int, default=256, help='Number of embeded time dimension')
  parser.add_argument('--number_noise_steps', type=int, default=1000, help='Numbe of steps required to noise the image')
  parser.add_argument('--beta_start', type=float, default=1e-4, help='First value of beta')
  parser.add_argument('--beta_end', type=float, default=0.02, help='Last value of beta')
  parser.add_argument('--noise_schedule', type=str, default="sigmoid", help='How the value of beta will change over time')
  parser.add_argument('--image_size', type=int, default=32, help='Size of the squared input image')
  parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
  parser.add_argument('--number_workers', type=int, default=2, help='Number of workers for the dataloader')
  parser.add_argument('--number_steps', type=int, default=200, help='How many iterations steps the model will learn from')
  parser.add_argument('--number_epochs', type=int, default=200, help='Number of epochs the model will learn from')
  parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate of the optmizer')
  parser.add_argument('--number_classes', type=int, default=10, help='Number of classes for the classifier')
  parser.add_argument('--checkpoint_directory', \
  type=str, default="/", help='')

  args = parser.parse_args(params)

  #Import the Mninst dataset for training and validation
  transforms = torchvision.transforms.Compose([                                           
      torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size          
      torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
      torchvision.transforms.ToTensor(), 
      torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
                                                        
  dataset_train = CIFAR10("/content/CIFAR10", download=True, train=True,transform=transforms)

  dataloader_train = DataLoader(dataset_train,args.batch_size)
  args = parser.parse_args(params)

  diffusion_model = UNET(args, 3,3,number_classes_input=args.number_classes).to(args.device)

  optmizer = torch.optim.Adam(diffusion_model.parameters(), lr=args.learning_rate)

  loss_mse = nn.MSELoss()
  
  #Doing the calculation for the number of iterations
  size_iterations = len(dataloader_train.dataset)/args.batch_size
  params_update = ['--number_steps', str(int(size_iterations))]
  params = params + params_update
  args = parser.parse_args(params)

  if args.use_checkpoints == "True":
    model_checkpoint = torch.load(args.checkpoint_directory + "/" + args.noise_schedule + "_" + "DiffusionModel.pt" )
  else:
    model_checkpoint = None
  answer = input("What action to take: ")

  if answer == "train":
    print("Number iterations: ", size_iterations)
    train(args, diffusion_model, dataloader_train, optmizer, loss_mse,model_checkpoint)

    
  elif answer == "sample":
    diffusion = basics(args, args.number_noise_steps, args.beta_start, args.beta_end, args.image_size, args.device)
    if model_checkpoint is not None:
      diffusion_model.load_state_dict(model_checkpoint['model_state_dict'])
    labels_to_predict = torch.tensor(3).to(args.device)
    image_sample = diffusion.sampling_image(diffusion_model, 1, labels_to_predict)
    image_sample1 = torch.squeeze(image_sample)

    trsmr = ToPILImage()
    img_pil1 = trsmr(image_sample1)
    display(img_pil1)

if __name__ == "__main__": 
  main(["--device", "cuda",
        "--checkpoint_directory", "checkpoints"])
