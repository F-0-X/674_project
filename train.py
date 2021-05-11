import argparse

import ocnn
import torch
from torch.backends import cudnn

from utils import load_modelnet40
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    raise Exception("We have to use cuda in this project")

def main(args):

    # create checkpoint folder

    # load data (we want 70% data for training 15% data for validation 15% of data for testing)
    train_points, train_labels, train_categories = load_modelnet40(train=True)
    num_of_train_data = len(train_points)


    # get our model
    model = ocnn.LeNet(args.depth, args.channel, args.nout)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=(120, 180, 240), gamma=0.1)


    # cudnn will optimize execution for our network
    cudnn.benchmark = True             # in hw4 our TA has this line of code

    if args.evaluate:
        print("\nEvaluation only")

        test_points, test_labels, test_categories = load_modelnet40(train=False)
        num_of_test_data = len(test_points)

        # load the best check point

        # test the data

        return

    # get data ready?
    n_points_train = int(args.train_split_ratio * num_of_train_data)
    full_indices = np.arange(num_of_train_data)
    np.random.shuffle(full_indices)
    train_indices = full_indices[:n_points_train]
    val_indices = full_indices[n_points_train:]
    # perform training

    # we have this line in hw4
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)

    for epoch in range(args.start_epoch, args.epochs):

        # train loss
        train_loss = train()
        # validation loss
        val_loss = val()
        # scheduler step
        scheduler.step()
        # save check point
        # print something to the console
        print("[train loss] " + str(train_loss))
        print("[val loss] " + str(val_loss))

def train(model, dataset, optimizer, args):
    model.train()



if __name__ == "__main__":



    parser = argparse.ArgumentParser(description='Octree classification')

    args = parser.parse_args()



    main(args)
