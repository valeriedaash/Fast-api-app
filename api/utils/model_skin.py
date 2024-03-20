import torch
from torchvision.models import resnet34
import torchvision.transforms as T

def class_id_to_label(i):
    '''
    Input int: class index
    Returns class name
    '''

    labels = {0: 'benign', 1: 'malignant'}
    return labels[i]

def load_model():
    '''
    Returns resnet model with IMAGENET weights
    '''
    model = resnet34()
    model.fc = torch.nn.Linear(512, 1)
    model.load_state_dict(torch.load('utils/model_skin.pt', map_location='cpu'))
    model.eval()
    return model

def transform_image(img):
    '''
    Input: PIL img
    Returns: transformed image
    '''
    trnsfrms = T.Compose(
        [
            T.Resize((224, 224)), 
            T.ToTensor(),
        ]
    )
    return trnsfrms(img)