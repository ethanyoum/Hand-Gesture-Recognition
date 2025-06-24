# Define the VGG model
model_vgg = models.vgg16(pretrained=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs_try , labels_try = inputs_try.to(device), labels_try.to(device)
model_vgg = model_vgg.to(device)
outputs_try = model_vgg(inputs_try)
outputs_try

print(model_vgg)

# Number of gesture classes
num_classes = len(full_dataset.lookup)
model_vgg.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
model_vgg = model_vgg.to(device)
