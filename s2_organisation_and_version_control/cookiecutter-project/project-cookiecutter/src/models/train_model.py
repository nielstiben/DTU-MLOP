# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from matplotlib import pyplot as plt
from torch import nn, optim


from model import MyAwesomeModel


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(exists=True))
def main(input_filepath, output_filepath):
    print("Training day and night")

    # Implement training loop here
    model = MyAwesomeModel()
    get_data = lambda filename: torch.load('{}/{}'.format(input_filepath, filename))
    train_X = get_data('train_X.pt')
    train_y = get_data('train_y.pt')
    train = []
    [train.append((single_train_X, train_y[ix])) for ix, single_train_X in enumerate(train_X)]
    trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 5
    loss_list = []
    for e in range(epochs):
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
    torch.save(model.state_dict(), '{}/trained_model.pt'.format(output_filepath))

    plt.plot(list(range(epochs)), loss_list)
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


