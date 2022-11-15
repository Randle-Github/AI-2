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

        meanAndStd = vaeEncoder(testImages)
        variationalPosterior = Normal(meanAndStd[:, 0], F.softplus(meanAndStd[:, 1]) + 1e-6)
        sample = variationalPosterior.rsample()

        xHat = vaeDecoder(sample)

        results = torch.cat([testImages, xHat], 0)
        results = torchvision.utils.make_grid(results, nrow=4)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        ax1.axis("off")
        ax1.imshow(results.permute(1, 2, 0))

        ax2.plot(losses)
        ax2.grid(True)
        plt.show()
    
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
        meanAndStd = vaeEncoder(x)
        # 基于均值和方差建立 变分后验分布
        variationalPosterior = Normal(meanAndStd[:, 0], F.softplus(meanAndStd[:, 1]) + 1e-6)
        # 先验分布，均值为 0，方差为 1
        prior = Normal(torch.zeros_like(meanAndStd[:, 0]), torch.ones_like(meanAndStd[:, 1]))
        # 先验分布和后验分布对齐
        klLoss = kl_divergence(variationalPosterior, prior).mean()
        # 从后验分布中采样
        sample = variationalPosterior.rsample()
        # 基于采样结果还原 x
        xHat = vaeDecoder(sample)

        loss = lossFn(xHat, x)
        # 重构损失 + 分布对齐
        loss = loss + 0.1 * klLoss

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
            visualizeVAE(vaeEncoder, vaeDecoder, testImgs, losses)
            vaeEncoder.train()
            vaeDecoder.train()
            torch.set_grad_enabled(True)
    print(f"Train on {step} steps finished.")
    print(f"Final loss: {loss.item()}")
    return

###AE
# 创建编码器和解码器
# encoder, decoder = Encoder(784, 200, 20), Decoder(20, 200, 28, 1)
# convEncoder, convDecoder = ConvEncoder(1, 200, 20), ConvDecoder(20, 200, 1)

# cifarencoder, cifardecoder = CifarEncoder(3, 200, 20), CifarDecoder(20, 200, 3)



# 使用 F.binary_cross_entropy 作为损失函数，学习率 0.0005，训练 6000 步
# train(encoder, decoder, mnistTraining, mnistTest, F.binary_cross_entropy, lr=0.0005, step=6000)

# train(cifarencoder, cifardecoder, cifarTraining, cifarTest, F.binary_cross_entropy, lr=0.001, step=1000)
vaeEncoder, vaeDecoder = VAEEncoder(1, 256, 16), VAEDecoder(16, 256, 1)

trainVAE(vaeEncoder, vaeDecoder, mnistTraining, mnistTest, F.binary_cross_entropy, lr=0.0005, step=2000)

''' 
###VAE
vaeEncoder = VAEEncoder(1, cHidden, k)

x = torch.rand(5, 1, 28, 28)

# [N, 2, k]
meanAndStd = vaeEncoder(x)

# Use torch.distributions.Normal
# Normal(mean, std) -> create a k-dim normal distribution
# mean <- meanAndStd[:, 0]
# std <- meanAndStd[:, 1]
variationalPosterior = Normal(meanAndStd[:, 0], F.softplus(meanAndStd[:, 1]) + 1e-6)

vaeDecoder = VAEDecoder(k, cHidden, 1)

sample = variationalPosterior.rsample()

vaeEncoder, vaeDecoder = VAEEncoder(1, cHidden, k), VAEDecoder(k, cHidden, 1)

trainVAE(vaeEncoder, vaeDecoder, mnistTraining, mnistTest, F.binary_cross_entropy, lr=0.0005, step=2000)
'''

