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