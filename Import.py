# Number of subprocesses to use for data loading
num_workers = 0
# How many samples per batch to load
batch_size = 64
# Percentage of training set to use as validation
train_size = 0.8
IMG_SIZE = 64

import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("gti-upm/leapgestrecog")

print("Path to dataset files:", path)

class LeapGestRecogDataset(Dataset):
    def __init__(self, data_path, transform=None, img_size=80):
        self.data_path = data_path
        self.transform = transform
        self.img_size = img_size
        self.x_data = []
        self.y_data = []
        self.lookup = {}
        self.reverselookup = {}
        self.label_count = 0

        # Create label mappings
        for folder_name in os.listdir(os.path.join(data_path, 'leapGestRecog', '00')):
            if not folder_name.startswith('.'):
                self.lookup[folder_name] = self.label_count
                self.reverselookup[self.label_count] = folder_name
                self.label_count += 1

        # Load images and labels
        for i in range(10):
            folder_path = os.path.join(data_path, 'leapGestRecog', f'0{i}')
            for gesture_folder in os.listdir(folder_path):
                if not gesture_folder.startswith('.'):
                    for image_file in os.listdir(os.path.join(folder_path, gesture_folder)):
                        img_path = os.path.join(folder_path, gesture_folder, image_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                        if img is None:
                            print(f"Warning: Image {img_path} could not be loaded.")
                            continue

                        # Resize and add to dataset
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        self.x_data.append(img)
                        self.y_data.append(self.lookup[gesture_folder])

        # Convert list to numpy array and normalize
        self.x_data = np.array(self.x_data, dtype='float32').reshape(-1, self.img_size, self.img_size, 1) / 255.0
        self.y_data = np.array(self.y_data)

        # Shuffle the dataset
        self.x_data, self.y_data = shuffle(self.x_data, self.y_data, random_state=42)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image = self.x_data[idx]
        label = self.y_data[idx]

        # Convert image to 3 channels to match expected input for some models
        image = np.repeat(image, 3, axis=-1)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Define transformations for the data
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create dataset
full_dataset = LeapGestRecogDataset(data_path=path, transform=data_transforms, img_size=IMG_SIZE)
