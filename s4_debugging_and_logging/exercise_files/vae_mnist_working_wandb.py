"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch
import wandb
from torch.optim import Adam
import argparse

hyperparameter_defaults = dict(
    batch_size = 100,
    epochs = 5,
)

def main():
    # Model Hyperparameters
    dataset_path = "datasets"
    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if cuda else "cpu")
    x_dim = 784
    hidden_dim = 400
    latent_dim = 20
    lr = 1e-3

    wandb.init(config=hyperparameter_defaults)
    config = wandb.config

    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(
        dataset_path, transform=mnist_transform, train=True, download=True
    )
    test_dataset = MNIST(
        dataset_path, transform=mnist_transform, train=False, download=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(Encoder, self).__init__()
            self.FC_input = nn.Linear(input_dim, hidden_dim)
            self.FC_mean = nn.Linear(hidden_dim, latent_dim)
            self.FC_var = nn.Linear(hidden_dim, latent_dim)
            self.training = True

        def forward(self, x):
            h_ = torch.relu(self.FC_input(x))
            mean = self.FC_mean(h_)
            log_var = self.FC_var(h_)
            std = torch.exp(0.5 * log_var)
            z = self.reparameterization(mean, std)
            return z, mean, log_var

        def reparameterization(
            self,
            mean,
            std,
        ):
            epsilon = torch.rand_like(std)
            z = mean + std * epsilon
            return z

    class Decoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim, output_dim):
            super(Decoder, self).__init__()
            self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
            self.FC_output = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h = torch.relu(self.FC_hidden(x))
            x_hat = torch.sigmoid(self.FC_output(h))
            return x_hat

    class Model(nn.Module):
        def __init__(self, Encoder, Decoder):
            super(Model, self).__init__()
            self.Encoder = Encoder
            self.Decoder = Decoder

        def forward(self, x):
            z, mean, log_var = self.Encoder(x)
            x_hat = self.Decoder(z)
            return x_hat, mean, log_var

    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    wandb.watch(model, log_freq=100)
    BCE_loss = nn.BCELoss()

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(
            x_hat, x, reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD


    optimizer = Adam(model.parameters(), lr=lr)
    print("Start training VAE...")
    model.train()
    for epoch in range(config.epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(config.batch_size, x_dim)
            x = x.to(DEVICE)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0: # report every 20 batches:
                wandb.log({"Loss": loss})
                model.eval()
                with torch.no_grad():
                    for batch_idx, (x, _) in enumerate(test_loader):
                        x = x.view(config.batch_size, x_dim)
                        x = x.to(DEVICE)
                        x_encoded,_,_ = encoder(x)
                        x_encoded_decoded = decoder(x_encoded) # Same as x_hat
                        noise = torch.randn(config.batch_size, latent_dim).to(DEVICE)
                        x_noise_decoded = decoder(noise)
                        break
                # Original
                wandb.log({'Original Images': wandb.Image(x.view(config.batch_size, 1, 28, 28))})
                # Encoded -> Decoded (x_hat)
                wandb.log({'Reconstructed Images (first Encoded, then Decoded)': wandb.Image(x_encoded_decoded.view(config.batch_size, 1, 28, 28))})
                # Decoded from random noice
                wandb.log({'Generated Images (random noise trough Decoder)': wandb.Image(x_noise_decoded.view(config.batch_size, 1, 28, 28))})
                model.train()
        else:
            print("\tEpoch",epoch + 1,"complete!","\tAverage Loss: ",overall_loss / (batch_idx * config.batch_size))
            wandb.log({"Average Loss over epochs": overall_loss / (batch_idx * config.batch_size)})


    print("Finish!!")
    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(config.batch_size, x_dim)
            x = x.to(DEVICE)
            x_hat, _, _ = model(x)
            break
    save_image(x.view(config.batch_size, 1, 28, 28), "orig_data.png")
    # wandb.log({'Original Images': wandb.Image(x.view(config.batch_size, 1, 28, 28))})
    save_image(x_hat.view(config.batch_size, 1, 28, 28), "reconstructions.png")
    # wandb.log({'Reconstructed Images': wandb.Image(x_hat.view(config.batch_size, 1, 28, 28))})
    # Generate samples
    with torch.no_grad():
        noise = torch.randn(config.batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)
    save_image(generated_images.view(config.batch_size, 1, 28, 28), "generated_sample.png")
    wandb.log({'Generated Images': wandb.Image(generated_images.view(config.batch_size, 1, 28, 28))})

if __name__ == "__main__":
    main()
