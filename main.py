from models.dpwgan import DPGAN
# from models.dpwGAN import DPGAN
from models.wGAN import WGAN
from utils.config import parse_args
from utils.data_loader import get_data_loader
import torch

def main(args):
    model = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model == "WGAN":
        model = WGAN(args)
        # model = model.to(device)
    elif args.model == "DPGAN":
        model = DPGAN(args)
        # model = model.to(device)

    train_loader, test_loader = get_data_loader(args)
    print("----------------------------------")
    print(type(args.is_train))
    if args.is_train == 'True':
        print("Let's start the training")
        model.train(train_loader=train_loader)
    
    else:
        print("What's Up")
        print(train_loader)

        print(args.load_discriminator, args.load_generator)
        # model.evaluate(test_loader, args.load_discriminator, args.load_generator)
        # print("What's Up")
        # for i in range(50):
        #     model.generate_latent_walk(i)

if __name__ == '__main__':
    args = parse_args()
    print("-^-"*20)
    print(args)
    # print(args.cuda)??
    main(args)