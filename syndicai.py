import torch
from PIL import Image 
from torchvision import transforms
import pickle
from torchvision import models
import torch.nn as nn
import zipfile

class PythonPredictor:

    def __init__(self, config):
        """ Download pretrained model. """
        print(torch.__version__)
        # Resnet101, paper uses 80 layer residual CNN.
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        # mpodify final layer
        model.fc = nn.Linear(num_ftrs, 300)
        print("Model loaded")
        
        # load model
        state_dict = torch.hub.load_state_dict_from_url(
            "https://www.googleapis.com/drive/v3/files/10c4gOh-BOF02Cu7FuXyEJGILYq6xK97n?alt=media&key=AIzaSyAzY7nf594k16gtOVrdFIKw66-7C3boBAc",
            map_location=torch.device('cpu')
        )
        model.load_state_dict(state_dict)
        print("state_dict updated")

        # get vectors
        with open('word_space.pkl', 'rb') as f:
            all_classes_df = pickle.load(f)
        print("Vectors loaded")

        self.classes = all_classes_df
        self.model = model


    def predict(self, payload):
        """ Run a model based on url input. """
        # open file
        input_image = Image.open(payload["url"])

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        data_point = self.model(input_batch)

        words = self.classes['tags'].tolist()
        vectors = torch.Tensor(self.classes['tags_vector'].tolist())

        dist = torch.norm(vectors - data_point, dim=1, p=None)
        knn = dist.topk(3, largest=False)

        predictions = []
        for val in knn.indices:
            label = words[val.item()]
            predictions.append(label)

        return predictions