class Config:
    pretrain = True
    posttrain = True
    pretrain_epochs = 100
    posttrain_epochs = 2000
    learning_rate = 0.001
    batch_size = 32
    mask_ratio = 0.1
    mask_value = 0
    train_split = 0.8
    hidden_dim = 64
    pf_dim = 128
    num_heads = 4
    num_layers = 1
    dropout_ratio = 0.1
    random_seed = 42
    data_path = './learning.csv'
    use_cuda = True
    save_folder = 'a_2test'
