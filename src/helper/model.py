from models.dymn.model import get_model as get_dymn
from src.config import settings
import torch


def load_model():
    model = get_dymn(pretrained_name=settings.MODEL_NAME)
    model.classifier = model.classifier[:4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model