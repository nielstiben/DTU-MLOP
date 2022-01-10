# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from matplotlib import pyplot as plt
from torch import nn, optim
import hydra
from omegaconf import OmegaConf

from model import MyAwesomeModel



@hydra.main(config_path="config", config_name='default.yaml')
def main(conf):
    print("Training day and night")
    print(os.getcwd())
    hparam = conf.hyperparameters
    def seed_everything(seed: int):
        import random, os
        import numpy as np
        import torch

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        param = conf.hyperparameters
        seed_everything(seed=param.seed)


    # Implement training loop here
    model = MyAwesomeModel(hparam.kernel_size)
    get_data = lambda filename: torch.load('../../../{}/{}'.format(conf.input_filepath, filename))
    train_X = get_data('train_X.pt')
    train_y = get_data('train_y.pt')
    train = []
    [train.append((single_train_X, train_y[ix])) for ix, single_train_X in enumerate(train_X)]

    trainloader = torch.utils.data.DataLoader(train, batch_size=hparam.batch_size, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparam.lr)

    loss_list = []
    for e in range(hparam.epochs):
        print('Epoch {}...'.format(e))
        running_loss = 0
        acc = []
        for images, labels in trainloader:
            # images = images.view(images.shape[0], -1)
            images = images.view(images.shape[0],1,images.shape[1],images.shape[2])

            optimizer.zero_grad()
            output = model(images)
            _,output_c = output.topk(1, dim=1)
            equals = output_c == labels.view(*output_c.shape)
            acc.append(torch.mean(equals.float()))

            loss = criterion(output, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        else:
            loss_list.append(running_loss/len(trainloader))
            print(f"Training loss: {running_loss/len(trainloader)} \t| Accuracy: {torch.mean(torch.stack(acc)).item()*100}%")

    # save model
    torch.save(model.state_dict(), '../../../{}/trained_model.pt'.format(conf.output_filepath))

    plt.plot(list(range(hparam.epochs)), loss_list)
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


