# -*- coding: utf-8 -*-

from torch.autograd import Variable
import torchvision.utils as vutils

from data import *
from models import *

from random import randint



def resume_file(resume_file):
    generator = Generator(tag_num=30)
    discriminator = Discriminator(30)
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        generator.load_state_dict(checkpoint['g_state_dict'])
        discriminator.load_state_dict(checkpoint['d_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_file, checkpoint['epoch']))
        return generator, discriminator
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))


def test(gen, dis):
    z = Variable(torch.FloatTensor(1, z_dim))
    tags = Variable(torch.FloatTensor(1, tag_num))

    z.data.normal_(0, 1)
    tags.data.bernoulli_(0.2)

    gen_img = gen(torch.cat((z, tags), 1))
    vutils.save_image(gen_img.data.view(1, 3, 128, 128), 'generate/{}.png'.format(randint(10, 99)))

def main():
    gen, dis = resume_file('models/Epoch029.checkpoint')
    test(gen, dis)

main()
