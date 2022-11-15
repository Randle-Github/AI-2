import torchvision.models as models
from torch import nn
import torch

# use a simple pre-trained ResNet50 to classify PACS
class resnet(nn.Module):
   def __init__(self, hidden_dim, num_classes):
      super(resnet, self).__init__()
      self.num_classes = num_classes
      self.resnet50 = models.resnet50(pretrained=True)
      self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, hidden_dim)
      self.linear2 = nn.Linear(hidden_dim, num_classes)
      self.relu = nn.ReLU(True)

   def forward(self, input):
      input = input.permute(0,3,1,2).cuda()
      x = self.resnet50(input)
      x = self.relu(x)
      x = self.linear2(x)
      return x
   
    
class TeacherNet():
   def __init__(self, file_path):
      super(TeacherNet, self).__init__()
      self.preds = torch.load(file_path)

   def get(self, idx):
      pred = []
      for i in range(len(idx)):
         pred.append(self.preds[int(idx[i])])
      return torch.stack(pred)