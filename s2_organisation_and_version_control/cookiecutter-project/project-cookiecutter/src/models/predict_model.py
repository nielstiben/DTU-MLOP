# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from matplotlib import pyplot as plt
from torch import nn, optim
from torchvision import transforms

import numpy as np

from model import MyAwesomeModel


@click.command()
@click.argument('trained_model_filepath', type=click.Path(exists=True))
@click.argument('data_filepath', type=click.Path(exists=True))
def main(trained_model_filepath, data_filepath):
    print("Predicting data")

    # load model
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(trained_model_filepath))
    model.eval()

    # load data
    imgs_np = np.load(data_filepath)
    imgs = torch.from_numpy(imgs_np).float().view(-1,1,28,28)
    transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
    imgs = transform(imgs)
    # predict
    _,output_c = model(imgs).topk(1, dim=1)
    filename = '{}_predictions.csv'.format(data_filepath.split('.')[0])
    np.savetxt(filename,output_c.flatten().numpy(),delimiter=',',fmt='%i')
    output_c.detach
    print('Done! Results saved to', filename)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


