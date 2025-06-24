# Define classes
classes = ['palm','I','fist','fist_moved','thumb','index','ok','palm_moved','c','down']
classes

# Visualize 10 samples from the training dataset
def visualize_training_data(dataset, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))

    for i in range(num_samples):
        # Get a sample image and label
        image, label = dataset[i]

        # Convert the tensor image to a numpy array
        image = image.numpy().transpose((1, 2, 0))

        # Display the original image
        image = image/2 + 0.5

        # Plot the image with label
        axes[i].imshow(image)
        axes[i].set_title(f'Label: {classes[label]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

visualize_training_data(train_dataset)
