def imshow(tensor, title=None):
    image = tensor.cpu().numpy().transpose((1, 2, 0))
    image = (image * 0.5) + 0.5
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Get the `lookup` from the dataset instance
lookup = full_dataset.lookup

# Create the dset_classes list from the lookup dictionary
dset_classes = [class_name for class_name in sorted(lookup.values())]

# Get a batch of validation data
inputs, classes = next(iter(test_loader))

# Set number of images to display
n_images = 10  # You can change this as needed

# Ensure we don't exceed the batch size
inputs_batch = inputs[:n_images]
classes_batch = classes[:n_images]

# Create a grid of images
out = torchvision.utils.make_grid(inputs_batch)

# Display the grid of images with class labels as the title
imshow(out, title=[dset_classes[x] for x in classes_batch])
