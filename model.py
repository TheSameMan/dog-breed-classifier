"""Dog breed classification model"""

from PIL import Image, UnidentifiedImageError
from torch.nn import BatchNorm1d, Linear
from torch import load
from torchvision.models import mobilenet_v3_large
from torchvision.transforms import (Compose, Resize, CenterCrop, ToTensor,
                                    Normalize)


class DogClassifier:
    """Neural network for predicting one of 133 classes of dog breeds"""
    def __init__(self):
        self.model = mobilenet_v3_large(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[2] = BatchNorm1d(1280)
        self.model.classifier[3] = Linear(1280, 133, bias=True)

        self.error = None
        try:
            with open('dogs', 'r') as file:
                self.names = file.read().split('\n')
        except FileNotFoundError:
            self.error = 'No dogs file'
        else:
            try:
                map_location = 'cpu'
                self.model.load_state_dict(
                    load('mobilenet_model.pt', map_location=map_location))

                self.model.eval()
            except FileNotFoundError:
                self.error = 'No model file'

    def __prepare_image(self, img_path):
        with Image.open(img_path, 'r') as image:
            tform = Compose([
                Resize(256),
                CenterCrop(227),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

            return tform(image).float().unsqueeze(0)

    def __call__(self, file_path):
        if (self.error == 'No dogs file') or (self.error == 'No model file'):
            return None

        self.error = None
        try:
            class_num = \
                self.model.forward(self.__prepare_image(file_path)).max(1)[1]
            return self.names[class_num]
        except (TypeError, UnidentifiedImageError):
            self.error = 'Wrong file format'
            return None

    def __repr__(self):
        return f'DogClassifier: {self.model.__repr__()}'
