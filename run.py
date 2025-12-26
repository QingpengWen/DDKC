# -*- coding: utf-8 -*-
"""
@CreateTime :       2024/05/30 21:25
@File       :       run.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2025/12/26 23:25
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error
from torch.optim import Adam
from tqdm import tqdm

from data.data_process import get_data, prepare_batches, data_level_out
from models.model_ddkc import DDKC
from utils.config import *

def run_training(train_data, val_data, test_data, test_df, model, optimizer,):
    # Training and validation on train & validation datasets

    while True:
        try:
            param_str = (f'{args.dataset},'
                         f'batch_size={args.batch_size},'
                         f'max_length={args.max_length},'
                         f'encode_pos={args.encode_pos},'
                         f'max_pos={args.max_pos}')
            logger = Logger(os.path.join(args.logdir, param_str))
            saver = Saver(args.savedir, param_str)
            if args.model_name == "ddkc":
                trains(train_data, val_data, test_data, model, optimizer, saver, args.num_epochs, args.batch_size, args.grad_clip)
            else:
                train(train_data, val_data, test_data, model, optimizer, saver, args.num_epochs, args.batch_size)
            break
        except RuntimeError:
            args.batch_size = args.batch_size // 2
            print(f'Batch does not fit on gpu, reducing size to {args.batch_size}')
    logger.close()

def run_evaluation(model, test_batches, test_df,):
    # Predict on test set
    test_preds = np.empty(0)
    mean_y = []
    model.eval()
    its = tqdm(test_batches, ncols=100)
    if args.model_name == "ddkc":
        for item_inputs, skill_inputs, label_inputs, d_count, d_correct, d_skill_correct, labels in its:  # Output as iter
            item_inputs = item_inputs.cuda()  # [batch, 200]
            skill_inputs = skill_inputs.cuda()  # [batch, 200]
            label_inputs = label_inputs.cuda()  # [batch, 200]
            d_count_inputs = d_count.cuda()  # [batch, 200]
            d_correct_inputs = d_correct.cuda()  # [batch, 200]
            d_skill_correct_inputs = d_skill_correct.cuda()  # [batch, 200]
            with torch.no_grad():
                preds = model(item_inputs, skill_inputs, label_inputs,
                              d_count_inputs, d_correct_inputs, d_skill_correct_inputs)[0]
                difficulty = model(item_inputs, skill_inputs, label_inputs,
                              d_count_inputs, d_correct_inputs, d_skill_correct_inputs)[4]
                preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
                test_preds = np.concatenate([test_preds, preds])
                mean_y.append(difficulty)
        print("Avg. Comprehensive Difficulty:", sum(mean_y) / len(mean_y))
    else:
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in its:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()
            with torch.no_grad():
                preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
                test_preds = np.concatenate([test_preds, preds])
    test_true = test_df["correct"]
    print("auc_test = ", roc_auc_score(test_true, test_preds))
    y_true = test_true
    y_pred = test_preds
    return y_true, y_pred

if __name__ == "__main__":
    # TODO: Datasets preparation
    full_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if args.model_name == "ddkc":
        if not os.path.exists(os.path.join(args.savedir, 'dataset.pkl')):
            if not os.path.exists(os.path.join('data', args.dataset, 'prelevel_data_test.csv')):
                print("Prelevel data is not exist, begin to preprocess test data.")
                test_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data_test.csv'), sep="\t")
                test_df = data_level_out(test_df, mode="test")
            else:
                test_df = pd.read_csv(os.path.join('data', args.dataset, 'prelevel_data_test.csv'), sep="\t")
                torch.save(test_df, os.path.join(args.savedir, 'dataset.pkl'))
        else:
            test_df = torch.load(os.path.join(args.savedir, "dataset.pkl"))
        test_data, _ = get_data(test_df, args.max_length, train_split=1.0, randomize=False)
    num_items = int(full_df["item_id"].max() + 1)
    num_skills = int(full_df["skill_id"].max() + 1)
    if args.model_name == "ddkc":
        model = DDKC(num_items, num_skills, args.embed_size, args.num_attn_layers, args.num_heads, args.encode_pos, args.max_pos, args.drop_prob).cuda()
    else:
        print("Please enter model name to training and evaluation")
        
    # TODO: Training
    if args.eval != "True":
        from train_ddkc import trains
        from utils import *
        print(args)
        if args.model_name == "ddkc":
            if not os.path.exists(os.path.join('data', args.dataset, 'prelevel_data_train.csv')):
                print("Prelevel data is not exist, begin to preprocess train data.")
                train_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data_train.csv'), sep="\t")
                train_df = data_level_out(train_df, mode="train", )
            else:
                train_df = pd.read_csv(os.path.join('data', args.dataset, 'prelevel_data_train.csv'), sep="\t")  # 'prelevel_data_train.csv'
        train_data, val_data = get_data(train_df, args.max_length)
        optimizer = Adam(model.parameters(), lr=args.lr)
        try:   
            run_training(train_data, val_data, test_data, test_df, model, optimizer)
        except KeyboardInterrupt:
            print("Exiting from training early.")
    
    # TODO: Evaluation
    print("Begin Evaluation: ")
    try:
        model.load_state_dict(torch.load(os.path.join('save', args.savedir, "model.pkl")))
    except FileNotFoundError:
        model.load_state_dict(torch.load(os.path.join('save', args.savedir, "model.pt")))
    test_batches = prepare_batches(test_data, args.batch_size, randomize=False)
    y_true, y_pred = run_evaluation(model, test_batches, test_df)

    print('\nAccepted test performance in ' + str(args.dataset) + " :")
    print("AUC:", roc_auc_score(y_true, y_pred))
    print("ACC:", accuracy_score(y_true, np.asarray(y_pred).round()), )
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", mean_squared_error(y_true, y_pred) ** 0.5)


