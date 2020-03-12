import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import xml.etree.ElementTree as ET
import os


def imshow(img, text=None, should_save=False):
  npimg = img.numpy()
  plt.axis("off")
  if text:
    plt.text(75, 8, text, style='italic', fontweight='bold',
             bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


def show_plot(iteration, loss):
  plt.plot(iteration, loss)
  plt.show()

class Config():
  #Data_path = "/home/smj/DataSet/ILSVRC/Data/VID/train/ILSVRC2015_VID_train_test/"
  image_root_path = "/home/smj/DataSet/ILSVRC/Data/VID/val/"
  xml_root_path = "/home/smj/DataSet/ILSVRC/Annotations/VID/val/"
  #image_root_path = "/Data_HDD/smj_data/ILSVRC/Data/VID/val/"
  #xml_root_path = "/Data_HDD/smj_data/ILSVRC/Annotations/VID/val/"
  #image_root_path = "/data/smj_data/DataSet/ILSVRC/Data/VID/val/"
  #xml_root_path = "/data/smj_data/DataSet/ILSVRC/Annotations/VID/val/"
  model_path = "./weight/model_resnet.pkl"
  train_batch_size = 48
  train_number_epochs = 100000

class SiameseNetworkDataset(Dataset):

  def __init__(self, image_data, xml_data, transform=None, should_invert=True):
    self.image_data = image_data
    self.xml_data = xml_data
    self.transform = transform
    self.should_invert = should_invert

  def __getitem__(self, index):


    while True:

      img0_class_id = random.randint(0, len(self.image_data)-1)

      img0_frame_id = random.randint(0, len(self.image_data[img0_class_id])-1)

      img0_frame_path = self.image_data[img0_class_id][img0_frame_id]
      img0_xml_path = self.xml_data[img0_class_id][img0_frame_id]

      img0_xml_tree = ET.parse(img0_xml_path)
      img0_xml_root = img0_xml_tree.getroot()

      try:
        img0_x1 = int(img0_xml_root[4][2][0].text)
        img0_x0 = int(img0_xml_root[4][2][1].text)
        img0_y1 = int(img0_xml_root[4][2][2].text)
        img0_y0 = int(img0_xml_root[4][2][3].text)
      except IndexError:
        continue

      img0_box = (img0_x0, img0_y0, img0_x1, img0_y1)

      break

    # we need to make sure approx 50% of images are in the same class
    should_get_same_class = random.randint(0, 1)
    # 0 or 1 (int)

    if should_get_same_class == 0:

      while True:
        img1_class_id = img0_class_id
        img1_frame_id = random.randint(0, len(self.image_data[img1_class_id]) - 1)

        if img0_frame_id == img1_frame_id:
          continue

        img1_frame_path = self.image_data[img0_class_id][img1_frame_id]
        img1_xml_path = self.xml_data[img0_class_id][img1_frame_id]

        img1_xml_tree = ET.parse(img1_xml_path)
        img1_xml_root = img1_xml_tree.getroot()

        try:
          img1_x1 = int(img1_xml_root[4][2][0].text)
          img1_x0 = int(img1_xml_root[4][2][1].text)
          img1_y1 = int(img1_xml_root[4][2][2].text)
          img1_y0 = int(img1_xml_root[4][2][3].text)
        except IndexError:
          continue

        img1_box = (img1_x0, img1_y0, img1_x1, img1_y1)

        break

    if should_get_same_class == 1:

      while True:

        img1_class_id = random.randint(0, len(self.image_data) - 1)

        img1_frame_id = random.randint(0, len(self.image_data[img1_class_id]) - 1)

        img1_frame_path = self.image_data[img1_class_id][img1_frame_id]
        img1_xml_path = self.xml_data[img1_class_id][img1_frame_id]

        img1_xml_tree = ET.parse(img1_xml_path)
        img1_xml_root = img1_xml_tree.getroot()

        try:
          img1_x1 = int(img1_xml_root[4][2][0].text)
          img1_x0 = int(img1_xml_root[4][2][1].text)
          img1_y1 = int(img1_xml_root[4][2][2].text)
          img1_y0 = int(img1_xml_root[4][2][3].text)
        except IndexError:
          continue

        img1_box = (img1_x0, img1_y0, img1_x1, img1_y1)
        break



    # 3 channel RGB
    img0 = Image.open(img0_frame_path)
    #img0 = img0.convert("L")
    img0 = img0.crop(img0_box)

    img1 = Image.open(img1_frame_path)
    #img1 = img1.convert("L")
    img1 = img1.crop(img1_box)

    if self.should_invert:
      img0 = PIL.ImageOps.invert(img0)
      img1 = PIL.ImageOps.invert(img1)

    if self.transform is not None:
      img0 = self.transform(img0)
      img1 = self.transform(img1)

    return img0, img1, torch.from_numpy(np.array([img1_class_id != img0_class_id], dtype=np.float32))

  def __len__(self):
    return len(self.image_data)





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


class ContrastiveLoss(torch.nn.Module):
  def __init__(self, margin=2.0):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def forward(self, output1, output2, label):
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

    return loss_contrastive





video_name_list = os.listdir(Config.image_root_path)
video_name_list.sort()

image_class_frame_path_list = []
xml_class_frame_path_list = []

for video_name_id in range(len(video_name_list)):
  video_name = video_name_list[video_name_id]

  image_video_path = os.path.join(Config.image_root_path, video_name)
  xml_video_path = os.path.join(Config.xml_root_path, video_name)

  image_class_frame_path_list.append([])
  xml_class_frame_path_list.append([])

  image_frame_list = os.listdir(image_video_path)
  image_frame_list.sort()

  xml_frame_list = os.listdir(xml_video_path)
  xml_frame_list.sort()

  for image_frame_name_id in range(len(image_frame_list)):
    image_frame_name = image_frame_list[image_frame_name_id]
    image_frame_path = os.path.join(image_video_path, image_frame_name)
    image_class_frame_path_list[video_name_id].append(image_frame_path)

  for xml_frame_name_id in range(len(xml_frame_list)):
    xml_frame_name = xml_frame_list[xml_frame_name_id]
    xml_frame_path = os.path.join(xml_video_path, xml_frame_name)
    xml_class_frame_path_list[video_name_id].append(xml_frame_path)



siamese_dataset = SiameseNetworkDataset(image_data=image_class_frame_path_list,
                                        xml_data=xml_class_frame_path_list,
                                        transform=transforms.Compose([transforms.Resize((224,224)),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                           std=[0.229, 0.224, 0.255])
                                                                      ]),
                                        should_invert=False)



net = torch.load(Config.model_path)
net = net.eval()

test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
#x0, _, _ = next(dataiter)

for i in range(20):
  x0, x1, label2 = next(dataiter)
  concatenated = torch.cat((x0, x1), 0)

  output1 = net(Variable(x0).cuda())
  output2 = net(Variable(x1).cuda())
  euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
  imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
  pass

