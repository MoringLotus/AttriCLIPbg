import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Sampler
from .imagenet100 import imagenet100
from .cifar100 import cifar100
import torchvision.datasets as dset
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import pdb

import collections
from utils.cutout import Cutout

class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, shuffle):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if(self.shuffle):
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
    

class IncrementalDataset:

    def __init__(
        self,
        dataset_name,
        args,
        random_order=False,
        shuffle=True,
        workers=8,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.
    ):
        self.dataset_name = dataset_name.lower().strip()
        datasets = _get_datasets(dataset_name)
        self.train_transforms = datasets[0].train_transforms 
        self.common_transforms = datasets[0].common_transforms
        try:
            self.meta_transforms = datasets[0].meta_transforms
        except:
            self.meta_transforms = datasets[0].train_transforms
        self.args = args
        
        self._setup_data(
            datasets,
            args.root,
            random_order=random_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split
        )
        

        self._current_task = 0

        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self.sample_per_task_testing = {}
    @property
    def n_tasks(self):
        return len(self.increments)
    
    def get_same_index(self, target, label, mode="train", memory=None):
        label_indices = []
        label_targets = []

        for i in range(len(target)):
            if int(target[i]) in label:
                label_indices.append(i)
                label_targets.append(target[i])
        for_memory = (label_indices.copy(),label_targets.copy())

            
        if memory is not None:
            memory_indices, memory_targets = memory
            memory_indices2 = np.tile(memory_indices, (1,))
            all_indices = np.concatenate([memory_indices2,label_indices])
        else:
            all_indices = label_indices
            
        return all_indices, for_memory
    
    def get_same_index_test_chunk(self, target, label, mode="test", memory=None):
        label_indices = []
        label_targets = []


        np_target = np.array(target, dtype="uint32")
        np_indices = np.array(list(range(len(target))), dtype="uint32")

        for t in range(len(label)//self.args.class_per_task):
            task_idx = []
            for class_id in label[t*self.args.class_per_task: (t+1)*self.args.class_per_task]:
                idx = np.where(np_target==class_id)[0]
                task_idx.extend(list(idx.ravel()))
            task_idx = np.array(task_idx, dtype="uint32")
            task_idx.ravel()
            random.shuffle(task_idx)

            label_indices.extend(list(np_indices[task_idx])) 
            label_targets.extend(list(np_target[task_idx]))
            if(t not in self.sample_per_task_testing.keys()):
                self.sample_per_task_testing[t] = len(task_idx)
        label_indices = np.array(label_indices, dtype="uint32")
        label_indices.ravel()
        return list(label_indices), label_targets
    

    def new_task(self, memory=None):
        print(self._current_task)
        print(self.increments)
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])

        
        train_indices, for_memory = self.get_same_index(self.train_dataset.targets, list(range(min_class, max_class)), mode="train", memory=memory)
        test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.targets, list(range(max_class)), mode="test")
        
        #class_name = self.train_dataset.get_classes()
        # 根据不同的数据集类型获取类别名称
        if self.dataset_name in ['imagenet100', 'cifar100']:
            class_name = self.train_dataset.get_classes()  # 这里使用 ImageNet100 和 CIFAR100 原来的 `get_classes`
        elif self.dataset_name == 'cifar100hd':
            class_name = self.train_dataset.classes  # 对于 ImageFolder 使用 `classes`
        elif self.dataset_name == 'cub':
            #class_name = self.train_dataset.classes  # 对于 ImageFolder 使用 `classes`
            class_name = [ #完整
            'Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet',
            'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird',
            'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting',
            'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat',
            'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant',
            'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow',
            'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch',
            'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher',
            'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar',
            'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe',
            'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak',
            'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull',
            'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull',
            'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear',
            'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay',
            'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher',
            'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark',
            'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser',
            'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole',
            'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican',
            'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will',
            'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx',
            'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow',
            'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow',
            'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow',
            'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 
            'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow',
            'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager',
            'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern',
            'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher',
            'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo',
            'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler',
            'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler',
            'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 
            'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler',  
            'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 
            'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', ###
            'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker',
            'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren',
            'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat'
        ]
    
        test_class = class_name[:max_class]
        class_name = class_name[min_class:max_class]
        

        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size,shuffle=False,num_workers=8, sampler=SubsetRandomSampler(train_indices, True))
        self.test_data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch,shuffle=False,num_workers=8, sampler=SubsetRandomSampler(test_indices, False))

        
        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(train_indices),
            "n_test_data": len(test_indices)
        }

        self._current_task += 1

        return task_info, self.train_data_loader, class_name, test_class, self.test_data_loader, self.test_data_loader, for_memory


     

    def get_galary(self, task, batch_size=10):
        indexes = []
        dict_ind = {}
        seen_classes = []
        for i, t in enumerate(self.train_dataset.targets):
            if not(t in seen_classes) and (t< (task+1)*self.args.class_per_task and (t>= (task)*self.args.class_per_task)):
                seen_classes.append(t)
                dict_ind[t] = i
                
        od = collections.OrderedDict(sorted(dict_ind.items()))
        for k, v in od.items(): 
            indexes.append(v)
            
        data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, False))
    
        return data_loader
    
    
    def get_custom_loader_idx(self, indexes, mode="train", batch_size=10, shuffle=True):
     
        if(mode=="train"):
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, True))
        else: 
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, False))
        return data_loader
    
    
    def get_custom_loader_class(self, class_id, mode="train", batch_size=10, shuffle=False):
        
        if(mode=="train"):
            train_indices, for_memory = self.get_same_index(self.train_dataset.targets, class_id, mode="train", memory=None)
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(train_indices, True))
        else: 
            test_indices, _ = self.get_same_index(self.test_dataset.targets, class_id, mode="test")
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(test_indices, False))
            
        return data_loader

    def _setup_data(self, datasets, path, random_order=False, seed=1, increment=10, validation_split=0.):
        self.increments = []
        self.class_order = []
        
        trsf_train = transforms.Compose(self.train_transforms)
        try:
            trsf_mata = transforms.Compose(self.meta_transforms)
        except:
            trsf_mata = transforms.Compose(self.train_transforms)
            
        trsf_test = transforms.Compose(self.common_transforms)
        
        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:
            if(self.dataset_name=="imagenet"or self.dataset_name=="imagenet100"):
                train_dataset = dataset.base_dataset(root=path, train=True, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, train=False, transform=trsf_test)
            elif(self.dataset_name=="cifar10" or self.dataset_name=="cifar100"):  # 修改：添加 CIFAR 数据集的选项
                train_dataset = dataset.base_dataset(root=path, train=True,  transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, train=False,  transform=trsf_test)
            elif self.dataset_name == "cifar100hd":
                train_dataset = dataset.get_base_dataset(root=path, train=True)
                test_dataset = dataset.get_base_dataset(root=path, train=False)
            elif self.dataset_name == "cub":
                train_dataset = dataset.get_base_dataset(root=path, train=True)
                test_dataset = dataset.get_base_dataset(root=path, train=False)
                # 按字母顺序排序 new_classes，并为每个类分配索引
                #sorted_classes = sorted(dataset.base_dataset.new_classes)  # 排序后的类
                #class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}  # 根据字母顺序分配新的索引
                # 创建新的 `order`：每个类别名按照字母顺序对应一个新的数字索引
                #order = [class_to_idx[cls] for cls in dataset.base_dataset.new_classes]
            
            order = [i for i in range(self.args.num_class)]
            if random_order:
                random.seed(seed)  
                random.shuffle(order)
            elif dataset.class_order is not None:
                order = dataset.class_order
                
            
            for i,t in enumerate(train_dataset.targets):
                train_dataset.targets[i] = order[t]
            for i,t in enumerate(test_dataset.targets):
                test_dataset.targets[i] = order[t]
            self.class_order.append(order)

            self.increments = [increment for _ in range(len(order) // increment)]

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))
    
    
    def get_memory(self, memory, for_memory, seed=1):
        random.seed(seed)
        memory_per_task = self.args.memory // ((self.args.sess+1)*self.args.class_per_task)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        mu = 1
        
        #update old memory
        if(memory is not None):
            data_memory, targets_memory = memory
            data_memory = np.array(data_memory, dtype="int32")
            targets_memory = np.array(targets_memory, dtype="int32")
            for class_idx in range(self.args.class_per_task*(self.args.sess)):
                idx = np.where(targets_memory==class_idx)[0][:memory_per_task]
                self._data_memory = np.concatenate([self._data_memory, np.tile(data_memory[idx], (mu,))   ])
                self._targets_memory = np.concatenate([self._targets_memory, np.tile(targets_memory[idx], (mu,))    ])
                
                
        #add new classes to the memory
        new_indices, new_targets = for_memory

        new_indices = np.array(new_indices, dtype="int32")
        new_targets = np.array(new_targets, dtype="int32")
        for class_idx in range(self.args.class_per_task*(self.args.sess),self.args.class_per_task*(1+self.args.sess)):
            idx = np.where(new_targets==class_idx)[0][:memory_per_task]
            self._data_memory = np.concatenate([self._data_memory, np.tile(new_indices[idx],(mu,))   ])
            self._targets_memory = np.concatenate([self._targets_memory, np.tile(new_targets[idx],(mu,))    ])
            
        print(len(self._data_memory))
        return list(self._data_memory.astype("int32")), list(self._targets_memory.astype("int32"))
    

def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "imagenet100":
        return iIMAGENET100()
    elif dataset_name == "cifar100":
        return iCIFAR100()  # 加载原始 CIFAR100
    elif dataset_name == "cifar100hd":
        return iCIFAR100HD()  # 加载超分辨率 CIFAR100 数据集
    elif dataset_name == "cub":
        return iCUB()  # 新添加的 CUB 数据集
    else:
        raise NotImplementedError(f"Unknown dataset {dataset_name}.")


class DataHandler:
    base_dataset = None
    train_transforms = []
    mata_transforms = [transforms.ToTensor()]
    common_transforms = [transforms.ToTensor()]
    class_order = None



class iIMAGENET100(DataHandler):
    base_dataset = imagenet100
    train_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
    ]
    common_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
    ]

class iCIFAR100(DataHandler):
    base_dataset = cifar100  # 使用 torchvision.datasets.CIFAR100
    
    train_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
    common_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        #transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]


from torchvision.datasets import ImageFolder

class iCIFAR100HD(DataHandler):
    """超分辨率后的 CIFAR-100 数据集"""
    
    base_dataset = ImageFolder  # 使用 ImageFolder 加载超分辨率图像
    
    train_transforms = transforms.Compose([  # 使用 transforms.Compose 组合转换
        transforms.Resize((224, 224), interpolation=BICUBIC),  
        lambda image: image.convert("RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    
    common_transforms = transforms.Compose([  # 使用 transforms.Compose 组合转换
        transforms.Resize((224, 224), interpolation=BICUBIC),  
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    def get_base_dataset(self, root, train=True):
        """加载超分辨率图像数据集"""
        folder = 'train' if train else 'test'
        dataset_path = os.path.join(root, 'cifar100_HD_images', folder)
        return ImageFolder(root=dataset_path, transform=self.train_transforms if train else self.common_transforms)


