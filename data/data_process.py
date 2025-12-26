# -*- coding: utf-8 -*-
"""
@CreateTime :       2024/05/28 21:25
@File       :       data_process.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2024/07/30 23:35
"""

import os
import torch
from random import shuffle
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from utils.config import *

def get_problemDefine(df, max_length, save_dir=None):
    
    input_df = df
    item_max_id = max(input_df["item_id"])
    item_min_id = min(input_df["item_id"])
    all_answer_count = len(input_df["item_id"])
    list_out = pd.DataFrame()
 
    # count for users
    for i in range(item_min_id, item_max_id + 1):
        list1 = input_df[input_df["item_id"].isin([i])]
        lens = len(list1["item_id"])
        if lens != 0:
            answer_count = len(list1["item_id"])  # count for answer
            if answer_count == 0:
                print("!!!Finding zero")
            d_answer_count = answer_count / all_answer_count
            list2 = list1[list1["correct"].isin([1])]
            list2 = list2["correct"].copy()
            sum_correct = sum(list2)  # count for right answer
            d_correct = sum_correct / answer_count 
            list3 = list1["skill_id"].copy()
            lenss = len(list1["skill_id"])
            d_skill_count = lenss / len(input_df["skill_id"])
            d_count = d_answer_count + d_skill_count

            skill_max_id = max(list3)
            skill_min_id = min(list3)
            if skill_max_id != skill_min_id:
                d_skill_correct = 0
                for i in range(skill_min_id, skill_max_id + 1):
                    list_skill = input_df[input_df["skill_id"].isin([i])]
                    lens_skill = len(list_skill["skill_id"])
                    if lens != 0:
                        list_skill_correct = list_skill[list_skill["correct"].isin([1])]
                        list_skill_correct = list_skill_correct["correct"].copy()
                        sum_skill_correct = sum(list_skill_correct) 
                        d_skill_correct = d_skill_correct + (sum_skill_correct / lens_skill)
            else:
                skill_id = skill_min_id
                list_skill = input_df[input_df["skill_id"].isin([skill_id])]
                lens_skill = len(list_skill["skill_id"])
                if lens != 0:
                    list_skill_correct = list_skill[list_skill["correct"].isin([1])]
                    list_skill_correct = list_skill_correct["correct"].copy()
                    sum_skill_correct = sum(list_skill_correct)  
                    d_skill_correct = sum_skill_correct / lens_skill  

            list1["d_count"] = d_count  
            list1["d_correct"] = d_correct  
            list1["d_skill_correct"] = d_skill_correct 
            
            list_out = list_out.append(list1)
            
    print("Finished List_out")
    max_d_count = max(list_out["d_count"])
    min_d_count = min(list_out["d_count"])
    list_out["d_count"] = (list_out["d_count"] - min_d_count) / (max_d_count - min_d_count)
    
    max_d_correct = max(list_out["d_correct"])
    min_d_correct = min(list_out["d_correct"])
    list_out["d_correct"] = (list_out["d_correct"] - min_d_correct) / (max_d_correct - min_d_correct)
    
    max_d_skill_correct = max(list_out["d_skill_correct"])
    min_d_skill_correct = min(list_out["d_skill_correct"])
    list_out["d_skill_correct"] = (list_out["d_skill_correct"] - min_d_skill_correct) / (max_d_skill_correct - min_d_skill_correct)
    return list_out

def get_data(df, max_length, train_split=0.8, randomize=True):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        max_length (int): maximum length of a sequence chunk
        train_split (float): proportion of data to use for training
    """

    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]
    d_count = [torch.tensor(u_df["d_count"].values, dtype=torch.float32)
               for _, u_df in df.groupby("user_id")]
    d_correct = [torch.tensor(u_df["d_correct"].values, dtype=torch.float32)
                 for _, u_df in df.groupby("user_id")]
    d_skill_correct = [torch.tensor(u_df["d_skill_correct"].values, dtype=torch.float32)
                       for _, u_df in df.groupby("user_id")]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    def chunk(list):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    # Chunk sequences
    lists = (item_inputs, skill_inputs, label_inputs, labels, d_count, d_correct, d_skill_correct)

    chunked_lists = [chunk(l) for l in lists]

    data = list(zip(*chunked_lists))
    if randomize:
        shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data

def prepare_batches(data, batch_size, randomize=True):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch
    Output:
        batches (list of lists of torch Tensor)
    """
    if randomize:
        shuffle(data)
    batches = []
    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))
        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          if (seqs[0] is not None) else None for seqs in seq_lists[:]]
        del inputs_and_ids[3]
        labels = pad_sequence(seq_lists[3], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs_and_ids, labels])

    return batches

def data_level_out(input_df, mode):
    
    data_defines = get_problemDefine(input_df, args.max_length, args.dataset)
    data_sorted_defines = data_defines.sort_index(ascending=True)   # sort by index
    data_path = os.path.join("data", args.dataset)
    
    if mode == "train":       
        data_sorted_defines.to_csv(os.path.join(data_path, "prelevel_data_train.csv"), sep="\t", 
                                   index=False)  # save as ".csv" file 
        data_sorted_defines.to_csv(os.path.join(data_path, "prelevel_data_train.txt"), sep='\t', 
                                   index=False)  # save as ".txt" file 
    elif mode == "test":
        data_sorted_defines.to_csv(os.path.join(data_path, "prelevel_data_test.csv"), sep="\t",
                                   index=False)  
        data_sorted_defines.to_csv(os.path.join(data_path, "prelevel_data_test.txt"), sep='\t', index=False) 
    else:
        print("Please enter mode")
        
    return data_sorted_defines

def format_list2str(input_list):
    return [str(x) for x in input_list]

def get_user_inters(df):
    """convert df to user sequences 

    Args:
        df (_type_): the merged df

    Returns:
        List: user_inters
    """
    user_inters = []
    for user, group in df.groupby("UserId", sort=False):
        group = group.sort_values(["answer_timestamp","tmp_index"], ascending=True)

        seq_skills = group['SubjectId_level3_str'].tolist()
        seq_ans = group['IsCorrect'].tolist()
        seq_response_cost = ["NA"]
        seq_start_time = group['answer_timestamp'].tolist()
        seq_problems = group['QuestionId'].tolist()
        seq_len = len(group)
        user_inters.append(
            [[str(user), str(seq_len)],
             format_list2str(seq_problems),
             format_list2str(seq_skills),
             format_list2str(seq_ans),
             format_list2str(seq_start_time),
             format_list2str(seq_response_cost)])
    return user_inters

