# predict.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import CustomModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path, model_path="dogvscat.pth", class_path="class_to_idx.pth"):
    model = CustomModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_to_idx = torch.load(class_path)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()

    label = idx_to_class[pred_idx]
    label = "Dog" if label.lower().startswith("dog") else "Cat"

    return label, confidence
