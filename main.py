import os
from Models.Autoencoder import Autoencoder
from Models.CBDNet import CBDNet
from Models.RIDNet import RIDNet
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import transforms
import numpy as np


def autoencoder():
    model = Autoencoder()

    model_id = "vaibhavprajapati22/Image_Denoising_Autoencoder"

    weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    return model


def cbdnet():
    model = CBDNet()

    model_id = "vaibhavprajapati22/Image_Denoising_CBDNet"

    weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    return model


def ridnet():
    model = RIDNet(3, 3, 128)

    model_id = "vaibhavprajapati22/Image_Denoising_RIDNet"

    weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    return model


def predict(image_path, model, save_path):
    model.eval()
    for file in os.listdir(image_path):
        path = os.path.join(image_path, file)
        image = Image.open(path)
        img_size = image.size
        image = image.resize((256, 256))
        input_transform = transforms.Compose([transforms.ToTensor()])
        image = input_transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            denoised_image = model(image)

        denoised_image = denoised_image.squeeze().detach().numpy()
        denoised_image = np.transpose(denoised_image, (1, 2, 0))

        denoised_image = (denoised_image * 255).astype(np.uint8)
        denoised_image_pil = Image.fromarray(denoised_image)
        denoised_image_pil = denoised_image_pil.resize(img_size)
        pred_path = os.path.join(save_path, file)
        denoised_image_pil.save(pred_path, format='PNG')


model = ridnet()
image_path = "test/low"
save_path = "test/predicted"
predict(image_path, model, save_path)
