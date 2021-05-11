import argparse
import torch
from torch.backends import cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    raise Exception("We have to use cuda in this project")


def main(args):

    # create checkpoint folder

    # load data (we want 70% data for training 15% data for validation 15% of data for testing)

    # get our model
    model = torch.nn.Module # change this line
    model.to(device)

    # cudnn will optimize execution for our network
    cudnn.benchmark = True             # in hw4 our TA has this line of code

    if args.evaluate:
        print("\nEvaluation only")

        # load the best check point

        # test the data

        return

    # get data ready?

    # perform training

    # we have this line in hw4
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)

    # for epoch in range(args.start_epoch, args.epochs):

        # train loss
        # validation loss
        # scheduler step
        # save check point
        # print something to the console



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Octree classification')