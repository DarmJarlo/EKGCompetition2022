# some training parameters
EPOCHS = 3
BATCH_SIZE = 128
NUM_CLASSES = 4
image_height = 90
image_width = 100
channels = 1
save_model_dir = "saved_model/model"
dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
Oned = True
# choose a network
# model = "resnet18"
# model = "resnet34"
model = "resnet50"
#model = "resnet_mini"
# model = "resnet101"
# model = "resnet152"
