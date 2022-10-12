import torch

@torch.no_grad()
def get_all_preds(model, loader, device, invert=False):
  all_preds = torch.tensor([]).to(device)
  for batch in loader:
    images, labels = batch
    if invert:
      images = 1 - images
    images, labels = images.to(device), labels.to(device)

    preds = model(images)
    all_preds = torch.cat((all_preds, preds) ,dim=0)

  return all_preds


def test(network, test_loader, device, output_file):
    network.eval()

    test_preds = get_all_preds(network, test_loader, device)
    actual_labels = torch.Tensor(test_set.targets).to(device)
    preds_correct = test_preds.argmax(dim=1).eq(actual_labels).sum().item()

    print('total correct:', preds_correct)
    with open(output_file, 'a') as the_file:
        the_file.write('total correct:', preds_correct)
    print('accuracy:', preds_correct / len(test_set))
    with open(output_file, 'a') as the_file:
        the_file.write('accuracy:', preds_correct / len(test_preds))
