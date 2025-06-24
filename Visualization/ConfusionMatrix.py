from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

num_classes = num_classes
predictions = predictions_CNN  # Predicted labels
all_classes = all_classes_CNN  # True labels

# Compute the confusion matrix
cm = confusion_matrix(all_classes, predictions, labels=range(num_classes))

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix for CNN Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()
