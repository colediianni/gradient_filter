import numpy as np
import torch
import torchvision
from torch import Tensor
from torchvision import transforms
import matplotlib.pyplot as plt


class GanDecolorizer(torch.nn.Module):
    def __init__(self, receptive_field: int = 1, distance_metric: str = "absolute"):
        super().__init__()
        self.receptive_field = receptive_field
        self.padding_val = torch.inf
        self.padding = torchvision.transforms.Pad(self.receptive_field, fill=self.padding_val, padding_mode='constant')
        self.distance_metric = distance_metric

    def forward(self, image: Tensor) -> Tensor:
        padded_input = self.padding(image)
        unfold = torch.nn.Unfold(kernel_size=(image.shape[2], image.shape[3]), padding=0, stride=1)
        inp_unf = unfold(padded_input)
        inp_unf = inp_unf.transpose(1, 2)
        inp_unf = inp_unf.reshape((image.shape[0], -1, 3, image.shape[2], image.shape[3]))
        inp_unf = inp_unf.permute([0, 1, 3, 4, 2])
        gradient_image = torch.zeros(image.shape[0], inp_unf.shape[1], inp_unf.shape[2], inp_unf.shape[3])
        image = image.permute([0, 2, 3, 1])

        for compare_shift in range(inp_unf.shape[1]):
            if self.distance_metric == "absolute":
                gradient_image[:, compare_shift, :, :] = torch.abs(torch.sub(image, inp_unf[:, compare_shift, :, :])).sum(dim=-1)
            elif self.distance_metric == "euclidean":
                gradient_image[:, compare_shift, :, :] = torch.sqrt(torch.square(torch.sub(image, inp_unf[:, compare_shift, :, :])).sum(dim=-1))

        gradient_image[gradient_image.isnan()] = 0
        gradient_image[gradient_image.isinf()] = 0
        gradient_image[gradient_image.abs() >= 10000] = 0

        return gradient_image



class remove_color(torch.nn.Module):
    def __init__(self, receptive_field: int = 1, distance_metric: str = "absolute", padding_val: float = torch.inf):
        super().__init__()
        self.receptive_field = receptive_field
        self.padding_val = padding_val
        self.padding = torchvision.transforms.Pad(self.receptive_field, fill=padding_val, padding_mode='constant')
        self.distance_metric = distance_metric

    def forward(self, image: Tensor) -> Tensor:
        padded_input = self.padding(image)
        unfold = torch.nn.Unfold(kernel_size=(image.shape[2], image.shape[3]), padding=0, stride=1)
        inp_unf = unfold(padded_input)
        inp_unf = inp_unf.transpose(1, 2)
        inp_unf = inp_unf.reshape((image.shape[0], -1, 3, image.shape[2], image.shape[3]))
        inp_unf = inp_unf.permute([0, 1, 3, 4, 2])
        gradient_image = torch.zeros(image.shape[0], inp_unf.shape[1], inp_unf.shape[2], inp_unf.shape[3])
        image = image.permute([0, 2, 3, 1])

        for compare_shift in range(inp_unf.shape[1]):
            if self.distance_metric == "absolute":
                gradient_image[:, compare_shift, :, :] = torch.abs(torch.sub(image, inp_unf[:, compare_shift, :, :])).sum(dim=-1)
            elif self.distance_metric == "euclidean":
                gradient_image[:, compare_shift, :, :] = torch.sqrt(torch.square(torch.sub(image, inp_unf[:, compare_shift, :, :])).sum(dim=-1))

        gradient_image[gradient_image.isnan()] = self.padding_val
        # gradient_image[gradient_image.isinf()] = 0
        # gradient_image = gradient_image.squeeze()

        return gradient_image

def remove_infs(image):
  image = image.clone()
  inf_indeces = image.isinf()
  image[inf_indeces] = 0
  # print(image.min(), image.max())
  return image.type(torch.int)


def colorize_gradient_image(original_image, device, bias_color_location=[], weighted=True, receptive_field=2, lr=1, squared_diff=True, image_is_rgb=True, verbose=False, num_iterations=500):

  original_image = original_image.clone()
  image_shape = original_image.shape
  image_shape = (image_shape[0], 3, image_shape[2], image_shape[3])
  # print(image_shape)

  if image_is_rgb:
    gradient_image = transforms.Compose([remove_color(receptive_field, "euclidean")])(original_image)
  else:
    gradient_image = original_image

  gradient_image = (gradient_image * 255).type(torch.int)
  gradient_image = gradient_image.clone().to(device)

  h, w = image_shape[2], image_shape[3]

  if len(bias_color_location) == 0:
    colorized_images = (torch.rand(image_shape)*255).type(torch.int).to(device)
  elif bias_color_location[1] == "all":
    # pass
    colorized_images = (torch.zeros(image_shape)).type(torch.int).to(device)
    colorized_images = colorized_images.permute([0, 2, 3, 1])
    # print(torch.tensor(bias_color_location[0]).type(torch.int).to(device))
    colorized_images += torch.tensor(bias_color_location[0]).type(torch.int).to(device)
    colorized_images = colorized_images.permute([0, 3, 1, 2])

  # remove same-pixel comparison from gradients
  num_directions = gradient_image.shape[1]
  center_pixel_value = int(num_directions/2)
  gradient_image = torch.cat([gradient_image[:, :center_pixel_value], gradient_image[:, center_pixel_value+1:]], dim=1)
  usable_gradients = torch.logical_and((gradient_image <= 255), (gradient_image >= 0))
  usable_gradients = usable_gradients*1

  padding = torchvision.transforms.Pad(receptive_field, padding_mode='reflect')

  for p in range(num_iterations):
    updated_colorized_images = colorized_images.detach().clone().type(torch.float).requires_grad_(requires_grad=True).to(device)
    updated_colorized_images = padding(updated_colorized_images)
    updated_colorized_images.retain_grad()

    if verbose:
        plt.imshow(remove_infs(colorized_images[0].permute([1, 2, 0])).cpu().detach().numpy())
        plt.show()

    diff_to_diff = torch.tensor(0, dtype=torch.float, requires_grad=True).to(device)
    # fill in with correct gradients
    for direction in range(num_directions-1):
      if direction >= center_pixel_value:
        neighbor_x_shift = (direction+1) % int(np.sqrt(num_directions))
        neighbor_y_shift = int((direction+1) / int(np.sqrt(num_directions)))
      else:
        neighbor_x_shift = direction % int(np.sqrt(num_directions))
        neighbor_y_shift = int(direction / int(np.sqrt(num_directions)))

      if weighted:
        weight = np.sqrt(np.square(neighbor_y_shift-receptive_field)+np.square(neighbor_x_shift-receptive_field))
      else:
        weight = 1

      predicted_gradients = torch.abs(updated_colorized_images[:, :, neighbor_y_shift:neighbor_y_shift+h, neighbor_x_shift:neighbor_x_shift+w] - updated_colorized_images[:, :, receptive_field:receptive_field+h, receptive_field:receptive_field+w]).permute([0, 2, 3, 1]).sum(dim=-1)

      # print("predicted_gradients", predicted_gradients.max())
      # print("gradient_image", gradient_image.max())
      if not squared_diff:
          diff_to_diff = diff_to_diff + (1/weight) * torch.mul(torch.abs(predicted_gradients - gradient_image[:, direction]), usable_gradients[:, direction]).sum()
      elif squared_diff:
          diff_to_diff = diff_to_diff + (1/weight) * torch.mul(torch.square(predicted_gradients - gradient_image[:, direction]), usable_gradients[:, direction]).sum()

    # print("diff_to_diff", diff_to_diff)
    # backpropogate
    diff_to_diff.backward()

    update = updated_colorized_images.grad
    # print(update.min(), update.max(), update.type(torch.float).mean())
    # add some stochasticity (so even if all gradients are 0, backprop will still go through)
    stochasticity = torch.round((torch.rand(update.shape)-0.5) * 2).type(torch.int).to(device)
    # print(stochasticity.min(), stochasticity.max(), stochasticity.type(torch.float).mean())
    update += stochasticity
    # print("update", update.max())
    # print("update", torch.abs(update).min(), torch.abs(update).max(), torch.abs(update).type(torch.float).mean())
    dynamic_lr = lr/torch.abs(update).type(torch.float).mean()

    # print("(dynamic_lr * update)", (dynamic_lr * update).min(), (dynamic_lr * update).max(), (dynamic_lr * update).type(torch.float).mean())
    updated_colorized_images = updated_colorized_images - (lr * update)
    updated_colorized_images = torch.clip(updated_colorized_images, 0, 255)
    # updated_colorized_images = torch.clip(torch.round(updated_colorized_images).type(torch.int), 0, 255)

    # update colorized_image to be center image of updated_colorized_images
    new_image = updated_colorized_images[:, :, receptive_field:receptive_field+h, receptive_field:receptive_field+w]
    colorized_images = new_image.detach()

  if verbose:
      plt.imshow(remove_infs(colorized_images[0].permute([1, 2, 0])).cpu().detach().numpy())
      plt.show()

  colorized_images = colorized_images.type(torch.float).to(device)
  return colorized_images / 255
