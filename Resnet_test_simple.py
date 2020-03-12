# calculate the similarity (distance) between two simple images

import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils

class Config():
  model_path = "./weight/model_resnet.pkl"
  bear1_path = "./test_images/bear1.jpg"
  bear2_path = "./test_images/bear2.jpg"
  boat1_path = "./test_images/boat1.jpg"
  boat2_path = "./test_images/boat2.jpg"
  man1_path = "./test_images/man1.jpg"
  man2_path = "./test_images/man2.jpg"


class SiameseNetwork(nn.Module):
  def __init__(self, resnet):
    super(SiameseNetwork, self).__init__()
    self.resnet_layer = nn.Sequential(*list(resnet.children())[:-1])
    self.fc1 = nn.Linear(2048, 128)
    self.fc2 = nn.Linear(128, 5)

  def forward(self, x):
    output = self.resnet_layer(x)
    output = output.view(-1, 2048)
    output = self.fc1(output)
    output = self.fc2(output)
    return output


net = torch.load(Config.model_path)
net = net.cuda()
net = net.eval()

normalization = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

######################################
### change the input images here  ####
######################################

img1_path = Config.bear1_path
img2_path = Config.bear2_path

######################################

img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

img1_norm = normalization(img1)
img2_norm = normalization(img2)

img1_norm.unsqueeze_(dim=0)
img2_norm.unsqueeze_(dim=0)

output1 = net(Variable(img1_norm).cuda())
output2 = net(Variable(img2_norm).cuda())
distance = F.pairwise_distance(output1, output2)

print("distance: " + str(distance))





