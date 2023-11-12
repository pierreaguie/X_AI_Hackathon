from dataloader import *
from models import *
from train_methods import *

import matplotlib as plt
import numpy as np
import torch


#Exemple d'exécution possible, les paths ne sont pas à jours.


def run(model, device, train_set, test_set, epoch, lr = 1e-3, model_path, results_path, fig_path):
    """"la fonction entraîne le modèle, enregistre la meilleure version (loss) et enregistre les courbes des loss et accuracy."""

    train_losses, test_losses, train_accuracies, test_accuracies, min_test_loss, acc = train_plot_all(model, device, dataloader, test_set, epoch, rate = 1e-3, path = path)
    file = open(results_path, 'a')
    file.write(f'test_loss: {min_test_loss}, test_acc: {acc}, model: {model_path}')
    file.close()
    plt.plot(np.arange(epoch), train_losses)
    plt.plot(np.arange(epoch), test_losses)
    plt.savefig(fig_path+'model_path' + '_loss.png')
    plt.show()

    plt.plot(np.arange(epoch), train_accuracies)
    plt.plot(np.arange(epoch), test_accuracies)
    plt.savefig(fig_path+'model_path' + '_acc.png')
    plt.show()
    
#Par exemple

path_resnet = '/kaggle/input/models/models/ResNetPretrained.pth'
model = ResNet(2, True, True, path = path_resnet)

transform = transforms.Compose([transforms.RandomRotation(180),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])

train_set, test_set = data_loader(dataset_path, transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch = 100

model_path = 'resnet_pretrained'
results_path = 'results_resnet_pretrained'
fig_path = 'curve_resnet_pretrained'


run(model, device, train_set, test_set, epoch, lr = 1e-3, model_path, results_path, fig_path)