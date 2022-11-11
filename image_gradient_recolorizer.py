import numpy as np
import torch
import torchvision
from torch import Tensor
from torchvision import transforms

class remove_color(torch.nn.Module):
    def __init__(self, receptive_field: int = 1, distance_metric: str = "absolute") -> Tensor:
        super().__init__()
        self.receptive_field = receptive_field
        self.padding = torchvision.transforms.Pad(self.receptive_field, fill=torch.inf, padding_mode='constant')
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

        gradient_image[gradient_image.isnan()] = torch.inf
        # gradient_image[gradient_image.isinf()] = 0
        gradient_image = gradient_image.squeeze()

        return gradient_image

def remove_infs(image):
  image = image.clone()
  inf_indeces = image.isinf()
  image[inf_indeces] = 0
  # print(image.min(), image.max())
  return image.type(torch.int)



def colorize_gradient_image(image_shape, original_image, device, bias_colors_list=[], weighted=True, receptive_field=2, lr=1, squared_diff=True):

  gradient_image = transforms.Compose([remove_color(4, "absolute")])(original_image)
  gradient_image = (gradient_image * 255).type(torch.int)
  gradient_image = gradient_image.clone().to(device)

  # remove same-pixel comparison from gradients
  num_directions = gradient_image.shape[1]
  center_pixel_value = int(num_directions/2)
  gradient_image = torch.cat([gradient_image[:, :center_pixel_value], gradient_image[:, center_pixel_value+1:]], dim=1)
  usable_gradients = torch.logical_and((gradient_image <= 255), (gradient_image >= 0))
  usable_gradients = usable_gradients*1

  h, w = image_shape[2], image_shape[3]

  colorized_images = (torch.rand(image_shape)*255).type(torch.int).to(device)
  padding = torchvision.transforms.Pad(receptive_field, padding_mode='reflect')

  for p in range(300):
    updated_colorized_images = colorized_images.clone().type(torch.float).requires_grad_(requires_grad=True).to(device)
    updated_colorized_images = padding(updated_colorized_images)
    updated_colorized_images.retain_grad()

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

      if not squared_diff:
          diff_to_diff += (1/weight) * torch.mul(torch.abs(predicted_gradients - gradient_image[:, direction]), usable_gradients[:, direction]).sum()
      if squared_diff:
          diff_to_diff += (1/weight) * torch.mul(torch.square(predicted_gradients - gradient_image[:, direction]), usable_gradients[:, direction]).sum()

    # backpropogate
    diff_to_diff.backward()
    updated_colorized_images = updated_colorized_images - (lr * updated_colorized_images.grad)
    updated_colorized_images = torch.clip(updated_colorized_images.type(torch.int), 0, 255)

    # update colorized_image to be center image of updated_colorized_images
    new_image = updated_colorized_images[:, :, receptive_field:receptive_field+h, receptive_field:receptive_field+w].type(torch.int)
    colorized_images = new_image

  plt.imshow(remove_infs(colorized_images[0].permute([1, 2, 0])).cpu().detach().numpy())
  plt.show()

  return colorized_images