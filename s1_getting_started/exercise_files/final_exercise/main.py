import argparse
import sys

import torch

from data import mnist
from model import MyAwesomeModel
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # Implement training loop here
        model = MyAwesomeModel()
        train,_ = mnist()
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
        torch.save(model.state_dict(), 'trained_model.pt')

        plt.plot(list(range(epochs)), loss_list)
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        model.eval()
        _, test_set = mnist()
        test_set_np = np.array(test_set, dtype=object)
        imgs, labels = torch.stack(list(test_set_np[:,0])),torch.Tensor(list(test_set_np[:,1]))
        # imgs = imgs.view(imgs.shape[0], -1)
        imgs = imgs.view(imgs.shape[0],1,imgs.shape[1],imgs.shape[2])

        _,output_c = model(imgs).topk(1, dim=1)

        equals = output_c == labels.view(*output_c.shape)
        acc = torch.mean(equals.float())
        print("Validation accuracy: {}%".format(round(acc.item(),3)*100))


if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    