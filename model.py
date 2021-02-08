"""Dog breed classifier"""

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image


class DogClassifier:
    with open('dogs', 'r') as f:
        names = f.read().split('\n')

    def __init__(self):
        self.model = models.resnet50(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Linear(2048, 133, bias=True)

        map_location = torch.device('cpu')

        self.model.load_state_dict(
            torch.load('dog_breed_classifier.pt', map_location=map_location))
        self.model.eval()

    def __prepare_image(self, img_path):
        image = Image.open(img_path, 'r')

        tform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

        return tform(image).float().unsqueeze(0)

    def predict(self, file_path):
        class_num = \
            self.model.forward(self.__prepare_image(file_path)).max(1)[1]
        return DogClassifier.names[class_num]
