# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import transforms
import numpy as np
import glob
import torch
import pickle


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # <OWN>
    transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
    train = []
    for train_npz in glob.iglob('{}/train_*.npz'.format(input_filepath)):
        data = np.load(train_npz, mmap_mode='r')
        np_imgs,np_labels = data.f.images, data.f.labels
        imgs = transform(torch.from_numpy(np_imgs)).float()
        for ix in range(len(np_labels)):
            train.append(( imgs[ix], np_labels[ix]))

    test = []
    data = np.load('{}/test.npz'.format(input_filepath), mmap_mode='r')
    np_imgs,np_labels = data.f.images, data.f.labels
    imgs = transform(torch.from_numpy(np_imgs)).float()
    for ix in range(len(np_imgs)):
        test.append((imgs[ix], np_labels[ix]))

    # Store data
    train_np = np.array(train, dtype=object)
    imgs, labels = torch.stack(list(train_np[:,0])),torch.Tensor(list(train_np[:,1])).long()
    torch.save(imgs, '{}/train_X.pt'.format(output_filepath))
    torch.save(labels, '{}/train_y.pt'.format(output_filepath))
    test_np = np.array(test, dtype=object)
    imgs, labels = torch.stack(list(test_np[:,0])),torch.Tensor(list(train_np[:,1])).long()
    torch.save(imgs, '{}/test_X.pt'.format(output_filepath))
    torch.save(labels, '{}/test_y.pt'.format(output_filepath))
    # </OWN>


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
