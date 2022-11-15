from heads import *

# Encoder
class Encoder(nn.Module):
    """ Vanilla encoder with 3 fc layers 
    Args:
            dIn (int): Dim of input, if `x.shape = [N, C, H, W]`, then `dIn = C*H*W`
        dHidden (int): Dim of intermediate FCs.
             dZ (int): Dim of z.
             
    Inputs:
        x (torch.FloatTensor): [N, C, H, W]. Input images.
        
    Outputs:
        torch.FloatTensor: [N, dZ]. Latent variable z.
    """
    def __init__(self, dIn: int, dHidden: int, dZ: int):
        super().__init__()
        # [fc -> relu -> bn] * 2 -> fc 
        self._net = nn.Sequential(*[
            # Convert [N, C, H, W] to [N, C*H*W]
            nn.Flatten(),
            nn.Linear(dIn, dHidden),
            nn.ReLU(),
            nn.BatchNorm1d(dHidden),
            nn.Linear(dHidden, dHidden),
            nn.ReLU(),
            nn.BatchNorm1d(dHidden),
            nn.Linear(dHidden, dZ)
        ])
        
    def forward(self, x):
        return self._net(x)

# Encoder
class Decoder(nn.Module):
    """ Vanilla decoder with 3 fc layers 
    Args:
             dZ (int): Dim of z.
        dHidden (int): Dim of intermediate FCs.
             hw (int): Height/width of x.
        channel (int): Channel num of x.
             
    Inputs:
        z (torch.FloatTensor): [N, dZ]. Input latent variable z.
        
    Outputs:
        torch.FloatTensor: [N, channel, hw, hw]. xHat, restored x.
    """
    def __init__(self, dZ: int, dHidden: int, hw: int, channel: int):
        super().__init__()
        # [fc -> relu -> bn] * 2 -> fc 
        self._net = nn.Sequential(*[
            nn.Linear(dZ, dHidden),
            nn.ReLU(),
            nn.BatchNorm1d(dHidden),
            nn.Linear(dHidden, dHidden),
            nn.ReLU(),
            nn.BatchNorm1d(dHidden),
            nn.Linear(dHidden, hw * hw),
            # Convert [N, C*H*W] to [N, C, H, W]
            nn.Unflatten(-1, (channel, hw, hw)),
            # Map output to [0, 1]
            nn.Sigmoid()
        ])
        
    def forward(self, z):
        return self._net(z)

# Convolutional Encoder
class ConvEncoder(nn.Module):
    """ Convolutional encoder with 3 conv layers 
    Args:
            cIn (int): Channel of x.
        cHidden (int): Channel of intermediate Convs.
             cZ (int): Channel of z.
             
    Inputs:
        x (torch.FloatTensor): [N, 1, 28, 28]. Input x.
        
    Outputs:
        torch.FloatTensor: [N, cZ, 7, 7]. Latent variable z.
    """
    def __init__(self, cIn: int, cHidden: int, cZ: int):
        super().__init__()
        # Build conv network
        # every conv block looks like:
        # ###################################
        # nn.Conv2d(cIn, cOut, kernel, stride, padding),
        # nn.LeakyReLU(),
        # nn.BatchNorm2d(cOut),
        # ###################################
        # and last conv is a bare `nn.Conv2d`
        self._net = nn.Sequential(*[
            # nn.Conv2d(cIn, cHidden, 3, 2, 1),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(cHidden),
            # ...
            # nn.Conv2d(cHidden, cZ, 3, 1, 1)
            
            nn.Conv2d(cIn, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cZ, 3, 1, 1)
        ])
        
    def forward(self, x):
        return self._net(x)

# Convolutional Encoder
class ConvDecoder(nn.Module):
    """ Convolutional decoder with 3 conv layers 
    Args:
             cZ (int): Channel of z.
        cHidden (int): Channel of intermediate Convs.
             cX (int): Channel of xHat.

    Inputs:
        x (torch.FloatTensor): [N, cZ, 28, 28]. Input latent variable z.
        
    Outputs:
        torch.FloatTensor: [N, cX, 28, 28]. Outpus xHat.
    """
    def __init__(self, cZ: int, cHidden: int, cX: int):
        super().__init__()
        # Build conv network
        # every conv block looks like:
        # ###################################
        # nn.ConvTranspose2d(cIn, cOut, kernel, stride, padding, outputPadding),
        # nn.LeakyReLU(),
        # nn.BatchNorm2d(cOut),
        # ###################################
        # and last conv is a bare `nn.ConvTranspose2d`
        self._net = nn.Sequential(*[
            # nn.ConvTranspose2d(cZ, cHidden, 3, 2, 1, 1),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(cHidden),
            # ...
            # nn.Conv2d(cHidden, cZ, 3, 1, 1, 1)
            
            nn.ConvTranspose2d(cZ, cHidden, 4, 1, 1, 0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 4, 2, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cX, 3, 2, 1, 1),
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        return self._net(x)

class CifarEncoder(nn.Module):
    """ Cifar encoder
    Args:
            cIn (int): Channel of x.
        cHidden (int): Channel of intermediate Convs.
             cZ (int): Channel of z.
             
    Inputs:
        x (torch.FloatTensor): [N, cIn, 32, 32]. Input x.
        
    Outputs:
        torch.FloatTensor: [N, cZ, ?, ?]. Latent variable z.
    """
    def __init__(self, cIn: int, cHidden: int, cZ: int):
        super().__init__()
        self._net = nn.Sequential(*[
            nn.Conv2d(cIn, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cZ, 3, 1, 1)
        ])
        
    def forward(self, x):
        return self._net(x) # (N, 20, 2, 2)

class CifarDecoder(nn.Module):
    """ Cifar decoder
    Args:
             cZ (int): Channel of z.
        cHidden (int): Channel of intermediate Convs.
             cX (int): Channel of xHat.
             
    Inputs:
        x (torch.FloatTensor): [N, C, ?, ?]. Input latent variable z.
        
    Outputs:
        torch.FloatTensor: [N, cX, 32, 32]. xHat.
    """
    def __init__(self, cZ: int, cHidden: int, cX: int):
        super().__init__()
        self._net = nn.Sequential(*[
            nn.ConvTranspose2d(cZ, cHidden, 4, 1, 1, 0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 5, 2, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cX, 3, 2, 1, 1),
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        return self._net(x)

class VAEEncoder(nn.Module):
    """ VAE encoder
    Args:
            cIn (int): Channel of x.
        cHidden (int): Channel of intermediate Convs.
              k (int): Final output, K-dim GMM.
             
    Inputs:
        x (torch.FloatTensor): [N, cIn, 28, 28]. Input x.
        
    Outputs:
        torch.FloatTensor: [N, 2, k]. k-(mean and variance)s of GMM.
    """
    def __init__(self, cIn: int, cHidden: int, k: int):
        super().__init__()
        self._net = nn.Sequential(*[
            nn.Conv2d(cIn, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            # [2, 2]
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Flatten(),
            nn.Linear(4 * cHidden, 2 * k)
        ])
        
    def forward(self, x):
        # x: [N, 1, 28, 28] -> [N, hidden, 7, 7] -> [N, 2 * k]
        # return [N, 2, k]
        return self._net(x).reshape(x.shape[0], 2, -1)
    
class VAEDecoder(nn.Module):
    """ VAE decoder
    Args:
            k (int): k-dim GMM.
        cHidden (int): Channel of intermediate Convs.
             cX (int): Channel of xHat.
             
    Inputs:
        z (torch.FloatTensor): [N, k]. k samples from the k-dim GMM.
        
    Outputs:
        torch.FloatTensor: [N, cX, 28, 28]. xHat.
    """
    def __init__(self, k: int, cHidden: int, cX: int):
        super().__init__()
        self._net = nn.Sequential(*[
            nn.Linear(k, 4 * cHidden),
            nn.Unflatten(-1, (cHidden, 2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 4, 1, 1, 0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 4, 2, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cX, 3, 2, 1, 1),
            nn.Sigmoid()
        ])
        
    def forward(self, z):
        return self._net(z)

class Cifar_VAEEncoder(nn.Module):
    """ VAE encoder
    Args:
            cIn (int): Channel of x.
        cHidden (int): Channel of intermediate Convs.
              k (int): Final output, K-dim GMM.
             
    Inputs:
        x (torch.FloatTensor): [N, cIn, 28, 28]. Input x.
        
    Outputs:
        torch.FloatTensor: [N, 2, k]. k-(mean and variance)s of GMM.
    """
    def __init__(self, cIn: int, cHidden: int, k: int):
        super().__init__()
        self._net = nn.Sequential(*[
            nn.Conv2d(cIn, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            # [2, 2]
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.Flatten(),
            nn.Linear(4 * cHidden, 2 * k)
        ])
        
    def forward(self, x):
        # x: [N, 1, 28, 28] -> [N, hidden, 7, 7] -> [N, 2 * k]
        # return [N, 2, k]
        return self._net(x).reshape(x.shape[0], 2, -1)
    
class Cifar_VAEDecoder(nn.Module):
    """ VAE decoder
    Args:
            k (int): k-dim GMM.
        cHidden (int): Channel of intermediate Convs.
             cX (int): Channel of xHat.
             
    Inputs:
        z (torch.FloatTensor): [N, k]. k samples from the k-dim GMM.
        
    Outputs:
        torch.FloatTensor: [N, cX, 28, 28]. xHat.
    """
    def __init__(self, k: int, cHidden: int, cX: int):
        super().__init__()
        self._net = nn.Sequential(*[
            nn.Linear(k, 4 * cHidden),
            nn.Unflatten(-1, (cHidden, 2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 4, 1, 1, 0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 5, 2, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cX, 3, 2, 1, 1),
            nn.Sigmoid()
        ])
        
    def forward(self, z):
        return self._net(z)

class Generator(nn.Module):
    """ GAN generator
    Args:
              k (int): Number of samples from the Gaussian distribution.
        cHidden (int): Channel of intermediate Convs.
           cOut (int): Channel of xHat.
             
    Inputs:
        n (int): Batch size.
        
    Outputs:
        torch.FloatTensor: [n, cX, 28, 28]. xHat.
    """
    def __init__(self, k: int, cHidden: int, cX: int):
        super().__init__()
        self._k = k
        self._net = nn.Sequential(*[
            nn.Linear(k, 4 * cHidden),
            nn.Unflatten(-1, (cHidden, 2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 4, 1, 1, 0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 4, 2, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cHidden, 3, 2, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(cHidden),
            nn.ConvTranspose2d(cHidden, cX, 3, 2, 1, 1),
            nn.Sigmoid()
        ])
        
    def forward(self, n):
        z = Normal(0.0, 1.0).sample((n, self._k)).cuda()
        return self._net(z)

class Discriminator(nn.Module):
    """ GAN discriminator
    Args:
            cIn (int): Channel of x.
        cHidden (int): Channel of intermediate Convs.
           cOut (int): Channel of xHat.
             
    Inputs:
        x (torch.FloatTensor): [N, cIn, 28, 28]. Input x.
        
    Outputs:
        torch.FloatTensor: [n, 1]. Probability of x is a real image or not.
    """
    def __init__(self, cIn: int, cHidden: int):
        super().__init__()
        self._net = nn.Sequential(*[
            nn.Conv2d(cIn, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            # [2, 2]
            nn.Conv2d(cHidden, cHidden, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4 * cHidden, 1),
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        return self._net(x)

#Predefined blocks

#Double convolution with Gelu activation
#The size of the input is going to be size of the output, except for the number of channels
class d_conv(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels = False, residual = False):    #Residuals define if there is going to be a residual connection from the input to the output
    super().__init__()

    self.residual = residual

    if not mid_channels:
      mid_channels = out_channels

    self.double_conv = nn.Sequential(
      nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
      nn.GroupNorm(1, mid_channels),
      nn.GELU(),
      nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.GroupNorm(1, out_channels),)
    
  def forward(self, x):
    if self.residual:
      output = F.gelu(x + self.double_conv(x))
    else:
      output = self.double_conv(x)
    return output

#Define the self attention block

class SA(nn.Module):    #Standard attention block
  def __init__(self, in_channels, dimension_input_image):
    super().__init__()
    self.in_channels = in_channels
    self.dimension_input_image = int(dimension_input_image)
    self.mha = nn.MultiheadAttention(in_channels, 4, batch_first=True)
    self.layer_norm = nn.LayerNorm([in_channels])

    self.feed_forward = nn.Sequential(
      nn.LayerNorm([in_channels]),
      nn.Linear(in_channels, in_channels),
      nn.GELU(),
      nn.Linear(in_channels, in_channels),
    )
  def forward(self, x):
    
    x = x.view(-1, self.in_channels, int(self.dimension_input_image**2)).swapaxes(1,2)
    
    norm_x = self.layer_norm(x)
    attention_value, _ = self.mha(norm_x, norm_x, norm_x)
    attention_value = attention_value + x
    attention_value = self.feed_forward(attention_value) + attention_value
    output = attention_value.swapaxes(2, 1).view(-1, self.in_channels, self.dimension_input_image, self.dimension_input_image)
    return output

#Defining Downsaple, midsample, and upsample blocks

class DS(nn.Module):
  def __init__(self, in_channels, out_channels, size_time_dimension = 256):
    super().__init__()
    self.pool_layer = nn.Sequential(   #Pooling layer with some convolutional blocks to change the number of channels and add the resitual
        nn.MaxPool2d(2),
        d_conv(in_channels, in_channels, residual=True),
        d_conv(in_channels, out_channels),
    )
    self.emb_layer = nn.Sequential(    #It is needed to change the dimensions of the time tensor in order to sum it to the output of the pooling layer
        nn.SiLU(),  #Activation function
        nn.Linear(size_time_dimension, out_channels)
    )
  def forward(self, x, time_tensor):
    out1 = self.pool_layer(x)
    out2 = self.emb_layer(time_tensor)[:, :, None, None].repeat(1,1,out1.shape[-2], out1.shape[-1]) #In this line we first match the number of time dimensions wiht the number of channels
                                                                                                    #Then this "[:, :, None, None].repeat(1,1,x.shape[-2], x.shape[-1])" part server to match the size of time tensor to the size of the outputx
    return (out1 + out2)


class US(nn.Module):
  def __init__(self, in_channels, out_channels, size_time_dimension = 256):
    super().__init__()

    self.up_block = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) #Block to upsample the resolution of the input

    self.conv_block = nn.Sequential(    #Convolutional block
        d_conv(in_channels, in_channels, residual=True),
        d_conv(in_channels, out_channels),
    )
  
    self.emb_layer = nn.Sequential(    #It is needed to change the dimensions of the time tensor in order to sum it to the output of the pooling layer
          nn.SiLU(),  #Activation function
          nn.Linear(size_time_dimension, out_channels),
      )
  def forward(self, x, skip_input, time_tensor):
    out1 = self.up_block(x)
    out2 = torch.cat([skip_input, out1], dim=1)
    out3 = self.conv_block(out2)
    out4 = self.emb_layer(time_tensor)[:, :, None, None].repeat(1,1,out3.shape[-2], out3.shape[-1]) #In this line we first match the number of time dimensions wiht the number of channels
                                                                                                    #Then this "[:, :, None, None].repeat(1,1,x.shape[-2], x.shape[-1])" part serves to match the size of time tensor to the size of the outputx
    return out3 + out4

class UNET(nn.Module):
  def __init__(self, args, in_channels = 3 , out_channels = 3, size_time_dimension = 256, number_classes_input = None, dimension_input_image = 32):
    super().__init__()
    self.args = args
    
    self.size_time_dimension = size_time_dimension
    self.dimension_input_image = dimension_input_image
    #Now we start the definition of the layers with the especific dimensions of the channels
    self.in_channel = d_conv(in_channels, 64)
    self.down1 = DS(64, 128,self.size_time_dimension)
    self.sa1 = SA(128, self.dimension_input_image/2)
    self.down2 = DS(128, 256,self.size_time_dimension)
    self.sa2 = SA(256, self.dimension_input_image/4)
    self.down3 = DS(256, 256,self.size_time_dimension)
    self.sa3 = SA(256, self.dimension_input_image/8)

    self.bot1 = d_conv(256, 512)
    self.bot2 = d_conv(512, 512)
    self.bot3 = d_conv(512, 256)

    self.up1 = US(512, 128,self.size_time_dimension)
    self.sa4 = SA(128, self.dimension_input_image/4)
    self.up2 = US(256, 64,self.size_time_dimension)
    self.sa5 = SA(64, self.dimension_input_image/2)
    self.up3 = US(128, 64,self.size_time_dimension)
    self.sa6 = SA(64, self.dimension_input_image)
    self.out_channel = nn.Conv2d(64, out_channels, kernel_size=1)

    if number_classes_input is not None:
      self.classification_embeding = nn.Embedding(number_classes_input, int(size_time_dimension))
  
  def forward(self, x_input, time_tensor, y_input):

        if y_input is not None:
          embeded_labels = self.classification_embeding(y_input)
          time_tensor = time_tensor + embeded_labels

        #The self attention blocks are not mandatory for small sized images
        x1 = self.in_channel(x_input)
        x2 = self.down1(x1, time_tensor)
        #x2 = self.sa1(x2)  
        x3 = self.down2(x2, time_tensor)
        #x3 = self.sa2(x3)
        x4 = self.down3(x3, time_tensor)
        #x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x5 = self.up1(x4, x3, time_tensor)
        #x5 = self.sa4(x5)
        x5 = self.up2(x5, x2, time_tensor)
        #x5 = self.sa5(x5)
        x5 = self.up3(x5, x1, time_tensor)
        #x5 = self.sa6(x5)
        output = self.out_channel(x5)
        return output
