from torchvision.models import ResNet18_Weights

# Define ResNet model
model_resnet = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
num_classes = 10

inputs_try , labels_try = inputs_try.to(device), labels_try.to(device)
model_resnet = model_resnet.to(device)
outputs_try = model_resnet(inputs_try)

# Specify loss function
criterion = nn.CrossEntropyLoss()

# Specify optimizer
optimizer_resnet = optim.Adam(model_resnet.parameters(), lr = 0.001)

resnet_train_losses, resnet_val_losses = train_model_with_loss_tracking(model_resnet, train_loader = train_loader,
                                                                        val_loader = test_loader,
                                                                        criterion = criterion, optimizer = optimizer_resnet,
                                                                        epochs = epochs)
