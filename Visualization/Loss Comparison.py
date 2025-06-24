plt.figure(figsize=(8, 6))

# Plot for Custom CNN
plt.plot(range(1, len(test_acc_CNN) + 1), test_acc_CNN, label='Custom CNN', color='blue', marker='o')

# Plot for ResNet18
plt.plot(range(1, len(test_acc_resnet) + 1), test_acc_resnet, label='ResNet18', color='orange', marker='x')

# Plot for Vgg16
plt.plot(range(1, len(test_acc_vgg) + 1), test_acc_vgg, label='VGG16', color='Green', marker='x')

plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Validation Accuracy', fontsize=14)
plt.title('Validation Accuracy per Epoch for Custom CNN, ResNet18 and VGG16', fontsize=16)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
