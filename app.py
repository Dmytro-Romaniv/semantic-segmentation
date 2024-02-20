import torch
from segmentation_models_pytorch import Unet
import cv2 as cv
import numpy as np
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

color_map = np.array(
    [
        [0, 0, 0],  # 0: background
        [128, 0, 0],  # 1: aeroplane
        [0, 128, 0],  # 2: bicycle
        [128, 128, 0],  # 3: bird
        [0, 0, 128],  # 4: boat
        [128, 0, 128],  # 5: bottle
        [0, 128, 128],  # 6: bus
        [128, 128, 128],  # 7: car
        [64, 0, 0],  # 8: cat
        [192, 0, 0],  # 9: chair
        [64, 128, 0],  # 10: cow
        [192, 128, 0],  # 11: dining table
        [64, 0, 128],  # 12: dog
        [192, 0, 128],  # 13: horse
        [64, 128, 128],  # 14: motorbike
        [192, 128, 128],  # 15: person
        [0, 64, 0],  # 16: potted plant
        [128, 64, 0],  # 17: sheep
        [0, 192, 0],  # 18: sofa
        [128, 192, 0],  # 19: train
        [0, 64, 128],  # 20: tv/monitor
    ]
)

model = Unet(encoder_name="resnet18", in_channels=3, classes=21, activation="softmax")
model = model.to(device)

model.load_state_dict(torch.load("./models/best.pt", map_location=device))

model.eval()


def decode_segmap(image, colors, nc=21):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = colors[l, 0]
        g[idx] = colors[l, 1]
        b[idx] = colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def predict(model, img):
    img = cv.resize(img, (256, 256))
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img = img.unsqueeze(0)
    img = img.to(device)
    pred = model(img)
    pred = pred.squeeze(0)
    pred = pred.argmax(0)
    pred = pred.detach().cpu().numpy()
    return pred


def process_image(image):
    pred = predict(model, image)
    pred = decode_segmap(pred, color_map)
    pred = cv.resize(pred, (image.shape[1], image.shape[0]))
    return cv.addWeighted(pred, 0.7, image, 1, 0)


iface = gr.Interface(
    fn=process_image,
    inputs="image",
    outputs="image",
    title="Semantic Segmentation Demo",
    description="Upload an image and the model will return a semantic segmentation mask.",
    examples=[[f"examples/example ({i}).jpg"] for i in range(1, 11)],
)

iface.launch()
