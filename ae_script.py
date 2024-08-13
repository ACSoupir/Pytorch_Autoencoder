import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

mnist_test = pd.read_csv("mnist_test.csv")
mnist_train = pd.read_csv("mnist_train.csv")

mnist_data = pd.concat([mnist_test, mnist_train])
del mnist_test
del mnist_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#separating data
X = mnist_data.drop(columns = ["label"])
Y = mnist_data.label

#features
X = np.stack([c.values for n, c in X.items()], axis=1).astype(np.float32)
X.min()
X.max()
X = X/255
X.max()
#response
Y = Y.to_numpy()

class mnistDataset(Dataset):
    #setup object
    def __init__(self, features, response):
        self.x = torch.from_numpy(features)
        self.y = torch.from_numpy(response)
    #define length function
    def __len__(self):
        return len(self.y)
    #definte getting the item
    def __getitem__(self, idx):
        f_i = self.x[idx]
        r_i = self.y[idx]
        return f_i, r_i
#create dataset
mnist_dataset = mnistDataset(X, Y)

#create dataloader
mnist_dataloader = DataLoader(dataset = mnist_dataset, batch_size=512, shuffle=True)

#defining the model architecture
class autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #building the encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(32, 16),
        )
        #build the decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(128, 784),
            torch.nn.Sigmoid()
        )
    #tell model how to use encoder and decoder
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = autoencoder()
model = model.to(device)
#specify loss function
loss_function = torch.nn.MSELoss()
#optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 3e-4,
                             weight_decay=1e-8)

#set up the training loops
losses = []
output = []
#how many epoch
epochs = 1000
#image, _ = next(iter(mnist_dataloader))
#loops
for epoch in range(epochs):
    for image, _ in mnist_dataloader:
        image = image.to(device)
        reconstructed = model(image)
        loss = loss_function(reconstructed, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #append losses
        losses.append(loss.item())
    print("epoch {0} \t loss: {1}".format(epoch, round(losses[-1], 3)))
    #output.append((epochs, image, reconstructed))


input_image = image.cpu()
input_image = input_image.reshape(-1, 28, 28)

output_image = reconstructed.cpu().detach().numpy()
output_image = output_image.reshape(-1, 28, 28)
#plt.imshow(input_image[3])
#plt.imshow(output_image[2])
which = 4
fig, (ax1, ax2) = plt.subplots(1, 2)
#fig.suptitle("Things")
ax1.imshow(input_image[which])
ax2.imshow(output_image[which])


plt.plot(losses)

model.encoder(image[2])

torch.save(model, "model.pth")
model2 = torch.load("model.pth")
model2 = model2.eval().to(device)
model2.encoder(image)[[0]]





















