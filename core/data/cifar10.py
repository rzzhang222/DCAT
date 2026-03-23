import torch
import torchvision
import torchvision.transforms as transforms

from .dataset import SemiSupervisedDataset

DATA_DESC = {
    'data': 'cifar10',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465],
    'std': [0.2023, 0.1994, 0.2010],
}

class SemiSupervisedCIFAR10(SemiSupervisedDataset):

    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'cifar10', 'Only semi-supervised cifar10 is supported. Please use correct dataset!'
        self.dataset = torchvision.datasets.CIFAR10(train=train, **kwargs)
        self.num_classes = DATA_DESC['num_classes']
        self.mean_std = (DATA_DESC['mean'], DATA_DESC['std'])


def load_cifar10(data_dir, logger, use_augmentation='none', use_consistency=False,
                  take_amount=4000, aux_take_amount=None, take_amount_seed = 1,
                  add_aux_labels=False, validation=False, pseudo_label_model=None,
                  aux_data_filename=None
                  ):

    test_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()
        ])

    train_dataset = SemiSupervisedCIFAR10(base_dataset='cifar10', root=data_dir, train=True, download=True,
                                            transform=train_transform,
                                            take_amount=take_amount,
                                            take_amount_seed=take_amount_seed,
                                            aux_take_amount=aux_take_amount,
                                            validation=validation,
                                            aux_data_filename=aux_data_filename,
                                            add_aux_labels=add_aux_labels,
                                            pseudo_label_model=pseudo_label_model,
                                            logger=logger
                                          )
    test_dataset = SemiSupervisedCIFAR10(base_dataset='cifar10', root=data_dir, train=False, download=True,
                                         transform=test_transform, logger=logger)
    if validation:
        val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, train_dataset.val_indices)
        return train_dataset, test_dataset, val_dataset

    return train_dataset, test_dataset, None

class twotransform():
    def __init__(self,transform_w,transform_s):
        self.transform_w=transform_w
        self.transform_s=transform_s
        
    def __call__(self,x):
        return [self.transform_w(x),self.transform_s(x)]
class threetransform():
    def __init__(self,transform_w,transform_s,transform_ss):
        self.transform_w=transform_w
        self.transform_s=transform_s
        self.transform_ss=transform_ss
        
    def __call__(self,x):
        return [self.transform_w(x),self.transform_s(x),self.transform_ss(x)]        
        
transform1 = transforms.Compose(
             [#transforms.TrivialAugmentWide(num_magnitude_bins=31),
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()])  
transform2 = transforms.Compose(
            [transforms.TrivialAugmentWide(num_magnitude_bins=31),
             transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5),
             transforms.ToTensor()])
             
transform3 = transforms.Compose([
    transforms.RandomResizedCrop((32,32), scale=(0.2, 1.),interpolation=transforms.InterpolationMode.BICUBIC,),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        #transforms.RandomApply([GaussianBlur()],p=0.5),
        transforms.ToTensor(),
        #transforms.Normalize(mean,std),
    ])               

transform_double=twotransform(transform1, transform2) 
transform_triple=threetransform(transform1,transform2,transform2)   
def load_cifar10double(data_dir, logger, use_augmentation='none', use_consistency=False,
                  take_amount=4000, aux_take_amount=None, take_amount_seed = 1,
                  add_aux_labels=False, validation=False, pseudo_label_model=None,
                  aux_data_filename=None
                  ):

    test_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        # train_transform = transforms.Compose(
        #     [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5),
        #      transforms.RandomRotation(15), transforms.ToTensor()])
        train_transform=transform_double
        
        
        
    else:
        train_transform = test_transform

    train_dataset = SemiSupervisedCIFAR10(base_dataset='cifar10', root=data_dir, train=True, download=True,
                                            transform=train_transform,
                                            take_amount=take_amount,
                                            take_amount_seed=take_amount_seed,
                                            aux_take_amount=aux_take_amount,
                                            validation=validation,
                                            aux_data_filename=aux_data_filename,
                                            add_aux_labels=add_aux_labels,
                                            pseudo_label_model=pseudo_label_model,
                                            logger=logger
                                          )
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                         transform=test_transform)
    if validation:
        val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, train_dataset.val_indices)
        return train_dataset, test_dataset, val_dataset

    return train_dataset, test_dataset, None
def load_cifar10triple(data_dir, logger, use_augmentation='none', use_consistency=False,
                  take_amount=4000, aux_take_amount=None, take_amount_seed = 1,
                  add_aux_labels=False, validation=False, pseudo_label_model=None,
                  aux_data_filename=None
                  ):

    test_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        # train_transform = transforms.Compose(
        #     [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5),
        #      transforms.RandomRotation(15), transforms.ToTensor()])
        train_transform=transform_triple
        
        
        
    else:
        train_transform = test_transform

    train_dataset = SemiSupervisedCIFAR10(base_dataset='cifar10', root=data_dir, train=True, download=True,
                                            transform=train_transform,
                                            take_amount=take_amount,
                                            take_amount_seed=take_amount_seed,
                                            aux_take_amount=aux_take_amount,
                                            validation=validation,
                                            aux_data_filename=aux_data_filename,
                                            add_aux_labels=add_aux_labels,
                                            pseudo_label_model=pseudo_label_model,
                                            logger=logger
                                          )
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                         transform=test_transform)
    if validation:
        val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, train_dataset.val_indices)
        return train_dataset, test_dataset, val_dataset

    return train_dataset, test_dataset, None    
    