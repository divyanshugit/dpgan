import torchvision.datasets  as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils


def get_data_loader(args):
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)),
        ])

        train_dataset = datasets.MNIST(root=args.dataroot, train=True, download=args.download, transform=transform)
        test_dataset = datasets.MNIST(root=args.dataroot, train=False, download=args.download, transform=transform)

    elif args.dataset == 'cifar':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),

        ])
        train_dataset = datasets.CIFAR10(root=args.dataroot, train=True, download=args.download, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.dataroot, train=False, download=args.download, transform=transform)

    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader
