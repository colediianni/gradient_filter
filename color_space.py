from torchvision import transforms
import cv2
from numpy import asarray


def to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)


def to_xyz(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


colorspaces = {
    "lab": [
        transforms.Lambda(asarray),
        transforms.Lambda(to_lab),
        transforms.ToTensor(),
    ],
    "xyz": [
        transforms.Lambda(asarray),
        transforms.Lambda(to_xyz),
        transforms.ToTensor(),
    ],
    "hsv": [
        transforms.Lambda(asarray),
        transforms.Lambda(to_hsv),
        transforms.ToTensor(),
    ],
    "rgb": [transforms.ToTensor()],
}
