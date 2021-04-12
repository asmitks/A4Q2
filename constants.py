trainFiles = {'Laptops':'ABSA complete Dataset/ABSA Train/Laptops_Train.xml','Restaurants':'ABSA complete Dataset/ABSA Train/Restaurants_Train.xml'}
testFiles = {'Laptops':'ABSA complete Dataset/ABSA Test/Laptops_Test_Gold.xml','Restaurants':'ABSA complete Dataset/ABSA Test/Restaurants_Test_Gold.xml'}

train_data = "train.csv"
test_data = "test.csv"

all_data = "data.csv"
embedding_size = 300
batch_size = 128
n_epoch = 10
hidden_size = 300
n_class = 3
pre_processed = True
learning_rate = 0.001
l2_reg = 0
clip = 3.0
dropout = 0.01
max_aspect_len = 21
max_context_len = 83
dataset = "data/restaurant/"
embedding_file_name = "../drive/MyDrive/glove/glove.840B.300d.txt"
embedding = 0
vocab_size = 0
embedding_dim = 300
embedding_size = 300
