from engine import *
from models import build_model
import argparse
import datetime
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler
from crowd_datasets import build_dataset

def get_args_parser():


    
    
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='./new_public_density_data',
                        help='path where the dataset is')
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')
    
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(args)

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()


    print("Loading_data...")

    loading_data = build_dataset(args=args)
    # create the training and valiation set
    train_set, val_set = loading_data(args.data_root)
    # create the sampler used during training
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    # the dataloader for training

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)
    
    # sampler_train = torch.utils.data.SequentialSampler(train_set)
    # data_loader_train = DataLoader(train_set, 1, sampler=sampler_train,
    #                                  drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)


    print("Loaded_data...")
    t1 = time.time()
    result = evaluate_crowd_no_overlap(model, data_loader_val, device)
    t2 = time.time()

    # mae.append(result[0])
    # mse.append(result[1])
    # print the evaluation results
    print('=======================================test=======================================')
    print("mae:", result[0], "mse:", result[1], "time:", t2 - t1)
    # with open(run_log_name, "a") as log_file:
    #     log_file.write("mae:{}, mse:{}, time:{}, best mae:{}".format(result[0], 
    #                     result[1], t2 - t1, np.min(mae)))
    print('=======================================test=======================================')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)