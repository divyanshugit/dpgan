import argparse
import os

def check_args(args):
    try:
        assert args.epochs >= 1
    except:
        print("Number of epochs must be larger than or equal to one")

    try: 
        assert args.batch_size >= 1
    except:
        print("Batch size must be larger than or equal to one")

    if args.dataset=='cifar' or args.dataset == 'stl10':
        args.channels = 3
    else:
        args.channels = 1
    
    return args

def parse_args():
    print("Is it okay??")
    parser = argparse.ArgumentParser(description="model: WGAN")

    parser.add_argument('--model', type=str, default='WGAN', choices=['WGAN', 'GAN', 'DPGAN'])
    # parser.add_argument('--is_train', type=str, default=True, choices=[True, False])
    parser.add_argument('--is_train', type=str, default='True')
    parser.add_argument('--dataroot', help="path to dataset")
    parser.add_argument('--download', type=bool, default=False, help="Select what you want to choose")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'celebA','CIFAR10'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64, help="The size of batch")
    parser.add_argument('--load_discriminator', type=str, default=False, help= 'Path for loading Discriminator Network')
    parser.add_argument('--load_generator', type=str, default=False, help= "Path for loading Generator network")
    parser.add_argument('--generator_iters', type=int, default=10000, help= "The number of iterations for generator in WGAN")
    
    return check_args(parser.parse_args())