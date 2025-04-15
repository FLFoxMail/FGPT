from datetime import datetime
import sys
class Config:
    def __init__(self):
        # 设置默认参数
        self.d_k = 128
        self.d_v = 128
        self.d_model = 128
        self.num_heads = 8
        self.d_diff = 1024
        self.n_layer = 2
        self.batch_size = 16
        self.seq_length = 1024
        self.data_path = "data/wiki_zh1"
        self.vocab_path = "check_points/wiki_zh1/vocab.txt"
        self.dataset = "SDataset"
        self.model_type = "SGPT"
        self.usage_percentage = 0.04
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        self.device = "cuda"
        self.eval_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 测试的次数（用于取平均值）
        self.num_rounds = 2
        # 循环次数，用于训练模型
        self.epochs = 2
        self.vocab_size = 0


    def parse_args(self):
        if len(sys.argv) == 12:
            self.d_k = int(sys.argv[1])
            self.d_v = int(sys.argv[2])
            self.d_model = int(sys.argv[3])
            self.num_heads = int(sys.argv[4])
            self.d_diff = int(sys.argv[5])
            self.n_layer = int(sys.argv[6])
            self.batch_size = int(sys.argv[7])
            self.seq_length = int(sys.argv[8])
            self.data_path = str(sys.argv[9])
            self.dataset = str(sys.argv[10])
            self.model_type = str(sys.argv[11])
        elif len(sys.argv) != 1:
            print("Usage: python model_benchmark.py [d_k d_v d_model num_heads d_diff n_layer batch_size seq_length data_path dataset model_type]")
            sys.exit(1)