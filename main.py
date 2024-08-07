import torch
import os
from model import FTTransformer, FTTransformerNew
from pretrain import pretrain
from posttrain import posttrain
from utils import save_parameters, load_data, set_seed, get_device
from dataset import TabularDataset, create_dataloaders
from experiments import experiments  # experiments.py 파일에서 설정을 가져옵니다.


def run_experiment(args):
    set_seed(args['random_seed'])
    device = get_device(args['use_cuda'])
    X_numerical, y, num_numerical_cols = load_data(args['data_path'])

    # 저장 폴더가 없으면 생성
    os.makedirs(args['save_folder'], exist_ok=True)

    # 데이터셋 및 DataLoader 생성
    dataset = TabularDataset(X_numerical, y, mask_ratio=args['mask_ratio'], mask_value=args['mask_value'])
    train_loader, test_loader = create_dataloaders(dataset, args['batch_size'], args['train_split'])

    if args['pretrain']:
        pretrain(args, train_loader, test_loader, num_numerical_cols, device, args['pretrain_epochs'],
                 args['pretrain_model_name'])

    if args['posttrain']:
        posttrain(args, train_loader, test_loader, num_numerical_cols, device, args['posttrain_epochs'],
                  args['posttrain_model_name'])

    save_parameters(args)


def main():
    for experiment in experiments:
        run_experiment(experiment)


if __name__ == "__main__":
    main()
