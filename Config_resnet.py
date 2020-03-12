import os

class Config_resnet():

  ########################################
  ###############  Training ##############
  ########################################

  # lr history 1e-6
  lr = 1e-6
  train_batch_size = 32
  train_number_epochs = 10000
  save_epochs = 1
  first_train = False

  # path to the model
  model_path = "./weight/model_resnet.pkl"

  # path to the trainig dataset
  root_path = "/home/smj/DataSet/ILSVRC/"
  #root_path = "/data/smj_data/DataSet/ILSVRC/"
  #root_path = "/Data_HDD/smj_data/ILSVRC"

  image_root_path = os.path.join(root_path, "Data/VID/train/ILSVRC2015_VID_train_0000/")
  xml_root_path = os.path.join(root_path, "Annotations/VID/train/ILSVRC2015_VID_train_0000/")

  image_root_path1 = os.path.join(root_path, "Data/VID/train/ILSVRC2015_VID_train_0001/")
  xml_root_path1 = os.path.join(root_path, "Annotations/VID/train/ILSVRC2015_VID_train_0001/")

  image_root_path2 = os.path.join(root_path, "Data/VID/train/ILSVRC2015_VID_train_0002/")
  xml_root_path2 = os.path.join(root_path, "Annotations/VID/train/ILSVRC2015_VID_train_0002/")

  image_root_path3 = os.path.join(root_path, "Data/VID/train/ILSVRC2015_VID_train_0003/")
  xml_root_path3 = os.path.join(root_path, "Annotations/VID/train/ILSVRC2015_VID_train_0003/")