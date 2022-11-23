# I'm sorry. Code needs to be cleaned up


from model import get_classification_model

base_path = "."
model_type="learned_diff_ci_resnet"
dataset_name="cifar"
colorspace="rgb"

model_save_path = (
    base_path+"/models/"+(model_type + "_" + dataset_name + "_" + colorspace + ".pth")
)

network = get_classification_model(
    model_type,
    device,
    3,
    load_from_path=model_save_path,
)

network.eval()
network = network.to(device)


from scipy.cluster.hierarchy import dendrogram, linkage

def get_rainbow_image(gradients_per_channel=range(255)):
  image = torch.zeros([len(gradients_per_channel)*len(gradients_per_channel)*len(gradients_per_channel), 3])
  index_to_add = 0
  for r in range(len(gradients_per_channel)):
    image[len(gradients_per_channel)*len(gradients_per_channel)*r:len(gradients_per_channel)*len(gradients_per_channel)*(r+1), 0] = gradients_per_channel[index_to_add]
    index_to_add += 1
  for g in range(len(gradients_per_channel)):
    sequence_to_add = torch.tensor(gradients_per_channel)
    sequence_to_add = sequence_to_add.unsqueeze(1)
    sequence_to_add = sequence_to_add.repeat(1, len(gradients_per_channel)).flatten()
    image[len(gradients_per_channel)*len(gradients_per_channel)*g:len(gradients_per_channel)*len(gradients_per_channel)*(g+1), 1] = sequence_to_add
  for b in range(len(gradients_per_channel)):
    sequence_to_add = torch.tensor(gradients_per_channel)
    sequence_to_add = sequence_to_add.repeat(len(gradients_per_channel))
    image[len(gradients_per_channel)*len(gradients_per_channel)*b:len(gradients_per_channel)*len(gradients_per_channel)*(b+1), 2] = sequence_to_add
  return image

image = get_rainbow_image(gradients_per_channel=range(0, 255, 10)).type(torch.float).to(device)
encoded_image = network.module.conv1.mapping_model(image.unsqueeze(1).permute([2, 0, 1]))
encoded_image = encoded_image.squeeze().permute([1, 0]).detach().cpu().numpy()

tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(encoded_image)

df = pd.DataFrame()
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

plt.scatter(df["comp-1"], df["comp-2"], c=image.detach().cpu().numpy()/255)
plt.show()




image = get_rainbow_image(gradients_per_channel=range(0, 255, 20)).type(torch.float).to(device)
encoded_image = network.module.conv1.mapping_model(image.unsqueeze(1).permute([2, 0, 1]))
encoded_image = encoded_image.squeeze().permute([1, 0]).detach().cpu().numpy()

# global count
# count = 0

def cs(x, y):
  # global count
  # print(count)
  # count += 1
  x, y = torch.tensor(x), torch.tensor(y)
  return 1 - torch.nn.CosineSimilarity(dim=0, eps=1e-08)(x, y)

# len(encoded_image)

import sys
sys.setrecursionlimit(10000)

Z = linkage(encoded_image, method="single", metric=cs)
Z = Z - Z.min()
fig = plt.figure(figsize=(40, 20))
dn = dendrogram(Z)
# dn = dendrogram(Z, labels=[str(x.tolist()) for x in image])
plt.show()


Z = linkage(encoded_image[1:], method="single", metric=cs)
Z = Z - Z.min()
fig = plt.figure(figsize=(40, 20))
dn = dendrogram(Z)
# dn = dendrogram(Z, labels=[str(x.tolist()) for x in image])
plt.show()
