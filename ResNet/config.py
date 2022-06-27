# some training parameters
EPOCHS = 10
BATCH_SIZE = 32
NUM_CLASSES = 4
image_height = 50
image_width = 180
channels = 1
save_model_dir = "saved_model/model"
dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"

# choose a network
# model = "resnet18"
# model = "resnet34"
model = "resnet50"
# model = "resnet101"
# model = "resnet152"
