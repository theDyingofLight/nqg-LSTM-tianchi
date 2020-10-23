file_path = r'/raid/wsy/comperition/nqg-paragraph/'
#file_path = r'F:/文本生成任务/competition-tianchi/nqg-paragraph/'
train_src_file = file_path + "dataset/para-train.txt"
train_trg_file = file_path + "dataset/tgt-train.txt"
train_ans_file = file_path + "dataset/ans-train.txt"

dev_src_file = file_path + "dataset/para-dev.txt"
dev_trg_file = file_path + "dataset/tgt-dev.txt"
dev_ans_file = file_path + "dataset/ans-dev.txt"

test_src_file = file_path + "dataset/para-test.txt"
test_trg_file = file_path + "dataset/tgt-test.txt"
test_ans_file = file_path + "dataset/ans-test.txt"

model_path = file_path + "save/model.pt"

device = "cuda:2"
use_gpu = True
debug = False

freeze_embedding = True

num_epochs = 20
max_len = 505
num_layers = 2
hidden_size = 768
embedding_size = 768
lr = 0.1
batch_size = 12
dropout = 0.4
max_grad_norm = 5.0

use_pointer = True
beam_size = 10
min_decode_step = 2
max_decode_step = 40
output_dir = "result/pointer_maxout_ans"
