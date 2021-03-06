# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from torch import nn
from model import MyAwesomeModel, get_activation, activation


@click.command()
@click.argument('trained_model_filepath', type=click.Path(exists=True))
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('image_ix_to_project')
@click.argument('projection_layer')
@click.argument('tsne_sample_limit')
def main(trained_model_filepath, data_filepath, image_ix_to_project, projection_layer, tsne_sample_limit):
    print("Visualizing data")

    # load pre-trained network
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(trained_model_filepath))
    model.eval()

    # load data
    model = MyAwesomeModel()
    get_data = lambda filename: torch.load('{}/{}'.format(data_filepath, filename))
    train_X = get_data('train_X.pt')
    train_y = get_data('train_y.pt')
    image_ix = int(image_ix_to_project)
    tsne_sample_limit = int(tsne_sample_limit)
    layer = projection_layer

    # Visulalize filter
    model_weights = []
    conv_layers = []
    for i, child in enumerate(model.children()):
        if type(child) == nn.Conv2d:
            model_weights.append(child.weight)
            conv_layers.append(child)
        elif type(child) == nn.Sequential:
            for j, child_child in enumerate(child.children()):
                if type(child_child) == nn.Conv2d:
                    model_weights.append(child_child.weight)
                    conv_layers.append(child_child)

    fig = plt.figure(figsize=(18, 4))
    nrows, ncols = 2, 8
    for i, filter in enumerate(model_weights[0]):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.imshow(filter[0, :, :].detach().numpy(), cmap='viridis')
        ax.axis('off')
        ax.set_title('filter {}'.format(i+1))
    plt.suptitle('Filters of {}'.format(layer))
    filename = "reports/figures/filter/filter_{}.png".format(layer)
    plt.savefig(filename)
    print("Filter visualisations of {} are saved as '{}'".format(layer, filename))
    plt.show()

    # Visualize feature maps
    img = train_X[image_ix].view(1, 1, 28, 28)
    getattr(model, layer).register_forward_hook(get_activation(layer)) # Forward hook
    _, output_c = model(img).topk(1, dim=1)
    feature_maps = activation[layer].squeeze()
    fig = plt.figure(figsize=(17, 14))
    nrows, ncols = 5, 7
    ax = fig.add_subplot(nrows, ncols, 1)
    ax.imshow(img.reshape(28, 28), interpolation='bilinear')
    ax.set_title('Original image')
    for i, feature_map in enumerate(feature_maps):
        ax = fig.add_subplot(nrows, ncols, i+2)
        ax.imshow(feature_map.numpy(), interpolation='bilinear')
        ax.set_title('feature map {}'.format(i+1))
    plt.suptitle("Feature maps of {} for image {}".format(layer, image_ix))
    filename = "reports/figures/feature_map/feature_map_{}_img{}.png".format(layer, image_ix)
    plt.savefig(filename)
    print("Feature maps of {} for image {} are saved as '{}'".format(layer, image_ix, filename))
    plt.show()

    # t-SNE
    limit = tsne_sample_limit
    getattr(model, layer).register_forward_hook(get_activation(layer)) # Forward hook
    _, output_c = model(train_X[:limit].view(-1, 1, 28, 28)).topk(1, dim=1)
    activations = activation.get(layer)

    tsne = TSNE(n_components=2)
    activations = tsne.fit_transform(activations.detach().numpy().reshape(activations.size(0), -1))

    sns.scatterplot(
        x=activations[:, 0], y=activations[:, 1], 
        hue=train_y[:limit].numpy(), 
        palette=sns.color_palette("hls", 10), 
        legend="full", 
        alpha=0.3
    )
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title("2D t-SNE representation of layer {}".format(layer))
    filename = "reports/figures/tSNE/tSNE_{}.png".format(layer)
    plt.savefig(filename)
    print("t-SNE maps of {} saved as '{}'".format(layer, filename))
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


