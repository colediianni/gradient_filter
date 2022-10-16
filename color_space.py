from torchvision import transforms
import cv2

colorspaces = {
    "lab":[transforms.Lambda(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2Lab)),
        transforms.ToTensor()],
    "xyz":[transforms.Lambda(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)),
        transforms.ToTensor()],
    "hsv":[transforms.Lambda(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV)),
        transforms.ToTensor()],
    "rgb":[transforms.ToTensor()]
}
