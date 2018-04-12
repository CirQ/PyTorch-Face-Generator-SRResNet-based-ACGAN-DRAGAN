# -*- coding: utf-8 -*-

import torch

IMAGE_PATH = '../illustrations_128'
TAG_PATH = 'tags.csv'

resume_file = ''
cuda = torch.cuda.is_available()
batch_size = 64
z_dim = 128
tag_num = 30
imsize = 128
start_epoch = 1
max_epochs = 200
lambda_adv = tag_num
lambda_gp = 0.5
learning_rate = 0.0002
