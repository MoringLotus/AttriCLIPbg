# cifar100.py
from types import new_class
from torchvision.datasets import CIFAR100
import os
import numpy as np

class cifar100(CIFAR100):

    cifar_templates = [
        'a photo of a {}.',
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]
    
    """new_classes = [
    'apple',           # 0
    'snail',           # 76
    'possum',          # 61
    'raccoon',         # 63
    'aquarium fish',   # 1
    'skyscraper',      # 71
    'baby',            # 2
    'bee',             # 6
    'can',             # 16
    'cattle',          # 19
    'bus',             # 13
    'cockroach',       # 24
    'mouse',           # 49
    'bridge',          # 12
    'worm',            # 75
    'bottle',          # 9
    'tulip',           # 83
    'skunk',           # 72
    'bed',             # 5
    'lion',            # 41
    'woman',           # 99
    'lobster',         # 45
    'whale',           # 89
    'palm tree',       # 53
    'sweet pepper',    # 79
    'castle',          # 18
    'pine tree',       # 52
    'wolf',            # 92
    'butterfly',       # 14
    'lizard',          # 42
    'streetcar',       # 68
    'leopard',         # 44
    'kangaroo',        # 38
    'turtle',          # 84
    'girl',            # 36
    'castle',          # 17
    'elephant',        # 31
    'camel',           # 15
    'ray',             # 70
    'woman',           # 88
    'couch',           # 25
    'wardrobe',        # 97
    'palm tree',       # 51
    'spider',          # 73
    'rose',            # 66
    'girl',            # 37
    'squirrel',        # 78
    'forest',          # 33
    'train',           # 80
    'crab',            # 26
    'trout',           # 82
    'cup',             # 28
    'plate',           # 60
    'fox',             # 35
    'lion',            # 43
    'mushroom',        # 57
    'cloud',           # 23
    'oak tree',        # 58
    'wolf',            # 91
    'bicycle',         # 8
    'possum',          # 62
    'worm',            # 93
    'wardrobe',        # 98
    'tank',            # 86
    'dinosaur',        # 29
    'dolphin',         # 30
    'chimpanzee',      # 22
    'turtle',          # 95
    'sunflower',       # 67
    'orange',          # 54
    'maple tree',      # 48
    'lawn mower',      # 40
    'oak tree',        # 59
    'tractor',         # 96
    'bear',            # 3
    'shrew',           # 87
    'fox',             # 34
    'rabbit',          # 64
    'mushroom',        # 56
    'snake',           # 69
    'man',             # 47
    'raccoon',         # 65
    'motorcycle',      # 50
    'television',      # 81
    'orange',          # 55
    'chair',           # 20
    'rose',            # 74
    'beaver',          # 4
    'willow tree',     # 90
    'crocodile',       # 27
    'telephone',       # 77
    'hamster',         # 32
    'house',           # 39
    'whale',           # 85
    'tiger',           # 94
    'chair',           # 21
    'lobster',         # 46
    'bowl',            # 10
    'boy',             # 11
    'beetle',          # 7
]"""

    def __init__(self, root, transform=None, train=True):
        split = 'train' if train else 'test'
        super(cifar100, self).__init__(root=root, train=train, transform=transform, download=True)

        #self.classes = self.new_classes
        
    def prompts(self, mode='single'):
        if mode == 'single':
            prompts = [[self.cifar_templates[0].format(label)] for label in self.classes]
            return prompts
        elif mode == 'ensemble':
            prompts = [[template.format(label) for template in self.cifar_templates] for label in self.classes]
            return prompts
    
    def get_labels(self):
        return np.array(self.targets)
    
    def get_classes(self):
        return self.classes
