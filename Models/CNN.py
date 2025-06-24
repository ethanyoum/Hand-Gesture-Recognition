class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(2, 2))
        self.conv1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.conv3_bn = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create a complete CNN
model_CNN = ImprovedNet()
model_CNN

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If using GPU, move the model to GPU
if torch.cuda.is_available():
    model_CNN = model_CNN.cuda()
else:
    model_CNN = model_CNN.to(device)

# Specify loss function
criterion = nn.CrossEntropyLoss()

# Specify optimizer
optimizer = optim.Adam(model_CNN.parameters(), lr=0.0001)

# Train
dset_sizes = {'trainval': len(train_dataset)}
model_cnn, training_loss, training_acc = train_model(model_CNN, train_loader, size=dset_sizes['trainval'], epochs=epochs, optimizer=optimizer)

# Test
dset_sizes = {'test': len(test_dataset)}
num_classes = len(full_dataset.lookup)
predictions_CNN, all_proba_CNN, all_classes_CNN, test_loss_CNN, test_acc_CNN = test_model(model_CNN,
test_loader, size=dset_sizes['test'], criterion=criterion, num_epochs = epochs,device=device)
