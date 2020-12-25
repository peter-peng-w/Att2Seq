word_dim = 512
dec_hid_dim = 512
enc_hid_dim = 64

rnn_layers = 2
dropout = 0.2
learning_rate = 2e-3
batch_size = 256
train_epochs = 50
REG_WEIGHT = 1e-4
CLIP = 5
train_batch = 4
val_batch = 64
test_batch = 64

# Data Process Parameters
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

MIN_FREQ = 2
MAX_LENGTH = 60

rating_range = 5
# rev_len = 31

dataset_path = './data/Musical_Instruments_5/'
# data_name = 'amazon_pet'
# target_path = '%s/data/%s' % (root_path, data_name)
# out_path = '%s/out/%s' % (root_path, data_name)
# model_path = '%s/out/model' % root_path
