from heads import *
from datasets import *
from model import *

lossFn = F.binary_cross_entropy
lossFn = F.mse_loss

############### 以下为训练用代码，请勿更改 ###############
def train(encoder: nn.Module, decoder: nn.Module, trainDataset, testDataset, lossFn, lr = 0.001, step = 1000):
    ############### 以下为可视化用代码，请勿更改 ###############
    def visualize(encoder: nn.Module, decoder: nn.Module, testImages: torch.Tensor, losses):
        # display.clear_output(wait=True)
        xHat = decoder(encoder(testImages))
        results = torch.cat([testImages, xHat], 0)
        results = torchvision.utils.make_grid(results, nrow=4)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        ax1.imshow(results.permute(1, 2, 0))
        ax1.axis("off")
        ax2.plot(losses)
        ax2.grid(True)
        plt.show()
    # 创建数据加载器
    loader = torch.utils.data.DataLoader(trainDataset, batch_size=8, shuffle=True, num_workers=0)
    # 创建网络优化器
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr)
    # 用于可视化
    testImgs = list()
    for i in range(4):
        testImgs.append(testDataset[i][0])
    testImgs = torch.stack(testImgs, 0)
    losses = list()

    iterator = iter(loader)
    
    for i in range(step):
        if i % 50 == 0:
            print(i)
        try:
            x, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, _ = next(iterator)
        # 编码过程
        z = encoder(x)
        # 解码过程
        xHat = decoder(z)
        # 计算重构损失
        loss = lossFn(xHat, x)
        # 记录 loss
        losses.append(loss.item())

        # 梯度回传训练网络
        optimizer.zero_grad(None)
        loss.backward()
        optimizer.step()

        # 可视化
        if i % (step - 1) == 0 and i != 0:
            print(float(loss))
            torch.set_grad_enabled(False)
            encoder.eval()
            decoder.eval()
            visualize(encoder, decoder, testImgs, losses)
            encoder.train()
            decoder.train()
            torch.set_grad_enabled(True)

    # 结束训练
    print(f"Train on {step} steps finished.")
    print(f"Final loss: {loss.item()}")
    return

def trainVAE(vaeEncoder: nn.Module, vaeDecoder: nn.Module, trainDataset, testDataset, lossFn, lr = 0.001, step = 1000):
    def visualizeVAE(vaeEncoder: nn.Module, vaeDecoder: nn.Module, testImages: torch.Tensor, losses):

        meanAndStd = vaeEncoder(testImages.cuda())
        variationalPosterior = Normal(meanAndStd[:, 0], F.softplus(meanAndStd[:, 1]) + 1e-6)
        sample = variationalPosterior.rsample()

        xHat = vaeDecoder(sample)

        results = torch.cat([testImages.cuda(), xHat], 0)
        results = torchvision.utils.make_grid(results, nrow=4)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        ax1.axis("off")
        ax1.imshow(results.permute(1, 2, 0).cpu())

        ax2.plot(losses)
        ax2.grid(True)
        plt.savefig("results.png")
    
    loader = torch.utils.data.DataLoader(trainDataset, batch_size=8, shuffle=True, num_workers=0)
    
    optimizer = torch.optim.Adam(list(vaeEncoder.parameters()) + list(vaeDecoder.parameters()), lr)
    
    testImgs = list()
    for i in range(4):
        testImgs.append(testDataset[i][0])
    testImgs = torch.stack(testImgs, 0)
    
    losses = list()

    iterator = iter(loader)
    
    for i in range(step):
        try:
            x, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, _ = next(iterator)
        # ###########################################
        # ############### VAE 训练过程 ###############
        # ###########################################
        # 从 x 预测 k 个高斯分布的均值和方差
        meanAndStd = vaeEncoder(x.cuda())
        # 基于均值和方差建立 变分后验分布
        variationalPosterior = Normal(meanAndStd[:, 0], F.softplus(meanAndStd[:, 1]) + 1e-6)
        # 先验分布，均值为 0，方差为 1
        prior = Normal(torch.zeros_like(meanAndStd[:, 0]), torch.ones_like(meanAndStd[:, 1]))
        # 先验分布和后验分布对齐
        klLoss = kl_divergence(variationalPosterior, prior).mean().cuda()
        # 从后验分布中采样
        sample = variationalPosterior.rsample().cuda()
        # 基于采样结果还原 x
        xHat = vaeDecoder(sample)

        loss = lossFn(xHat, x.cuda())
        # 重构损失 + 分布对齐
        loss = loss + 1.0 * klLoss

        losses.append(loss.item())

        optimizer.zero_grad(None)
        loss.backward()
        optimizer.step()

        if i % 50 == 0 :
            print(i, float(loss))
        if i % (step-1) == 0 and i !=0 :
            torch.set_grad_enabled(False)
            vaeEncoder.eval()
            vaeDecoder.eval()
            # visualizeVAE(vaeEncoder, vaeDecoder, testImgs, losses)
            vaeEncoder.train()
            vaeDecoder.train()
            torch.set_grad_enabled(True)
    print(f"Train on {step} steps finished.")
    print(f"Final loss: {loss.item()}")
    return

def trainGAN(generator: nn.Module, discriminator: nn.Module, trainDataset, testDataset, lr = 0.001, step = 1000):
    def visualizeGAN(generator: nn.Module, testImages: torch.Tensor, dLosses, gLosses):

        xHat = generator(testImages.shape[0])

        results = torch.cat([testImages, xHat.cpu()], 0)
        results = torchvision.utils.make_grid(results, nrow=4)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,5))
        ax1.imshow(results.permute(1, 2, 0).cpu())
        ax1.axis("off")

        ax2.plot(dLosses)
        ax2.grid(True)
        ax3.plot(gLosses)
        ax3.grid(True)
        plt.savefig("results.png")
    
    loader = torch.utils.data.DataLoader(trainDataset, batch_size=8, shuffle=True, num_workers=2)
    
    optimizerG = torch.optim.Adam(generator.parameters(), lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr)
    
    testImgs = list()
    for i in range(4):
        testImgs.append(testDataset[i][0])
    testImgs = torch.stack(testImgs, 0)
    
    dLosses = list()
    gLosses = list()
    
    iterator = iter(loader)
    
    for i in range(step):
        try:
            x, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, _ = next(iterator)
        x = x.cuda()
        # ###########################################
        # ############### GAN 训练过程 ###############
        # ###########################################
        # 生成一批假图片
        fake = generator(x.shape[0])
        # 判别过程
        if i % 2 == 0:
            realAndFake = torch.cat([x, fake], 0)
            probs = discriminator(realAndFake)
            realProb, fakeProb = torch.chunk(probs, 2)
            dLoss = F.binary_cross_entropy(realProb, torch.ones_like(realProb)) + F.binary_cross_entropy(fakeProb, torch.zeros_like(fakeProb))
            dLosses.append(dLoss.item())

            optimizerD.zero_grad(None)
            dLoss.backward()
            optimizerD.step()
        # 生成过程
        else:
            # 假图片输入判别器，生成假图片的置信度
            fakeProb = discriminator(fake)
            # 损失反转，让假图片的置信度提高
            gLoss = F.binary_cross_entropy(fakeProb, torch.ones_like(fakeProb))
            gLosses.append(gLoss.item())

            optimizerG.zero_grad(None)
            gLoss.backward()
            optimizerG.step()

        if i % 100 == 0 and i !=0 :
            print(i, "dLoss", float(dLoss))
        if i % 100 == 1 and i !=1:
            print(i, "gLoss", float(gLoss))

        if i % (step-1) == 0 and i != 0:
            torch.set_grad_enabled(False)
            generator.eval()
            generator.eval()
            visualizeGAN(generator, testImgs, dLosses, gLosses)
            generator.train()
            generator.train()
            torch.set_grad_enabled(True)

    print(f"Train on {step} steps finished.")
    return


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
    # display(img_pil1)
    epoch += 1
   
    
    #Saving Checkpoint
    

    EPOCH = epoch
    PATH = args.checkpoint_directory + "/" + "DiffusionModel.pt"      
    torch.save({
        'epoch': EPOCH,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optmizer.state_dict(),
        'LRSeg': args.learning_rate,
        }, PATH)
    print("checkpoint saved")

    print("The average loss was: ", np.mean(list_losses))

def ddpnmain(params):
  parser = argparse.ArgumentParser(description='Diffusion model')

  parser.add_argument('--device', type=str, default="cuda", help='Device to run the code on')
  parser.add_argument('--use_checkpoints', type=str, default="True", help='Use checkpoints')
  parser.add_argument('--emb_dimension', type=int, default=256, help='Number of embeded time dimension')
  parser.add_argument('--number_noise_steps', type=int, default=500, help='Numbe of steps required to noise the image')
  parser.add_argument('--beta_start', type=float, default=1e-4, help='First value of beta')
  parser.add_argument('--beta_end', type=float, default=0.02, help='Last value of beta')
  parser.add_argument('--noise_schedule', type=str, default="sigmoid", help='How the value of beta will change over time')
  parser.add_argument('--image_size', type=int, default=32, help='Size of the squared input image')
  parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
  parser.add_argument('--number_workers', type=int, default=2, help='Number of workers for the dataloader')
  parser.add_argument('--number_steps', type=int, default=50, help='How many iterations steps the model will learn from')
  parser.add_argument('--number_epochs', type=int, default=100, help='Number of epochs the model will learn from')
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
                                                        
  dataset_train = cifarTraining

  dataloader_train = DataLoader(dataset_train,args.batch_size)
  args = parser.parse_args(params)

  diffusion_model = UNET(args, 3,3, number_classes_input=args.number_classes).to(args.device)

  optmizer = torch.optim.Adam(diffusion_model.parameters(), lr=args.learning_rate)

  loss_mse = nn.MSELoss()
  
  #Doing the calculation for the number of iterations
  size_iterations = len(dataloader_train.dataset)/args.batch_size
  params_update = ['--number_steps', str(int(size_iterations))]
  params = params + params_update
  args = parser.parse_args(params)

  if args.use_checkpoints == "True":
    model_checkpoint = torch.load(args.checkpoint_directory + "/" + "DiffusionModel.pt" )
  else:
    model_checkpoint = None

  if False:
    print("Number iterations: ", size_iterations)
    train(args, diffusion_model, dataloader_train, optmizer, loss_mse,model_checkpoint)

    
  if True:
    diffusion = basics(args, args.number_noise_steps, args.beta_start, args.beta_end, args.image_size, args.device)
    if model_checkpoint is not None:
      diffusion_model.load_state_dict(model_checkpoint['model_state_dict'])
    labels_to_predict = torch.tensor(3).to(args.device)
    image_sample = diffusion.sampling_image(diffusion_model, 1, labels_to_predict)
    image_sample1 = torch.squeeze(image_sample).permute(1,2,0)

    # trsmr = ToPILImage()
    # img_pil1 = trsmr(image_sample1)
    cv2.imwrite("results.png", np.array(image_sample1.cpu()))

if __name__ == "__main__":

  ###AE
  '''
  # 创建编码器和解码器
  encoder, decoder = Encoder(784, 200, 20), Decoder(20, 200, 28, 1)
  convEncoder, convDecoder = ConvEncoder(1, 200, 20), ConvDecoder(20, 200, 1)

  cifarencoder, cifardecoder = CifarEncoder(3, 200, 20), CifarDecoder(20, 200, 3)



  # 使用 F.binary_cross_entropy 作为损失函数，学习率 0.0005，训练 6000 步
  train(encoder, decoder, mnistTraining, mnistTest, F.binary_cross_entropy, lr=0.0005, step=6000)

  train(cifarencoder, cifardecoder, cifarTraining, cifarTest, F.binary_cross_entropy, lr=0.001, step=1000)
  '''

  ###VAE
  '''
  vaeEncoder, vaeDecoder = VAEEncoder(1, 256, 16).cuda(), VAEDecoder(16, 256, 1).cuda()
  trainVAE(vaeEncoder, vaeDecoder, mnistTraining, mnistTest, F.binary_cross_entropy, lr=0.0005, step=4000)

  vaeEncoder, vaeDecoder = Cifar_VAEEncoder(3, 256, 64).cuda(), Cifar_VAEDecoder(64, 256, 3).cuda()
  trainVAE(vaeEncoder, vaeDecoder, cifarTraining, cifarTest, F.binary_cross_entropy, lr=0.0005, step=4000)

  z = Normal(0.0, 1.0).sample((5, 64))
  xHat = vaeDecoder(z.cuda())
  results = torchvision.utils.make_grid(xHat.cpu(), nrow=5)
  plt.imshow(results.permute(1, 2, 0).cpu())
  plt.axis("off")
  plt.savefig("results.png")
  '''

  ###GAN
  '''
  generator, discriminator = Generator(128, 128, 1), Discriminator(1, 128)
  generator = generator.cuda()
  discriminator = discriminator.cuda()
  trainGAN(generator, discriminator, mnistTraining, mnistTest, lr=0.0002, step=15000)
  '''

  ### ddpn
  ddpnmain(["--device", "cuda",
          "--batch_size", "8",
          "--checkpoint_directory", "/opt/data/private/liuyangcen/homework/work4/checkpoints"])
