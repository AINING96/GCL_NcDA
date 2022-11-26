import csv
import torch
import random
import numpy as np

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def data_pro(args):
    dataset = dict()
    dataset['md_p'] = read_csv(args.dataset_path + '/mi-d.csv')
    dataset['md_true'] = read_csv(args.dataset_path + '/mi-d.csv')

    zero_index = []
    one_index = []
    for i in range(dataset['md_p'].shape[0]):  # miRNA
        for j in range(dataset['md_p'].shape[1]):  # disease
            if dataset['md_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                one_index.append([i, j])
    random.shuffle(one_index)
    random.shuffle(zero_index)
    zero_tensor = torch.LongTensor(zero_index)
    one_tensor = torch.LongTensor(one_index)
    dataset['md'] = dict()
    dataset['md']['train'] = [one_tensor, zero_tensor]

    "disease GIP sim"
    dd_f_matrix = read_csv(args.dataset_path + '/d_d_GIP.csv')
    dd_f_edge_index = get_edge_index(dd_f_matrix)
    dataset['dd_g'] = {'data_matrix': dd_f_matrix, 'edges': dd_f_edge_index}

    "disease semantic sim"
    dd_s_matrix = read_csv(args.dataset_path + '/d_d_semantic.csv')
    dd_s_edge_index = get_edge_index(dd_s_matrix)
    dataset['dd_s'] = {'data_matrix': dd_s_matrix, 'edges': dd_s_edge_index}

    "disease DisGeNET sim"
    dd_s_matrix = read_csv(args.dataset_path + '/d_d_jaccard.csv')
    dd_s_edge_index = get_edge_index(dd_s_matrix)
    dataset['dd_h'] = {'data_matrix': dd_s_matrix, 'edges': dd_s_edge_index}

    # "disease hamads/ sim"
    # dd_s_matrix = read_csv(args.dataset_path + '/d_d_hamads.csv')
    # dd_s_edge_index = get_edge_index(dd_s_matrix)
    # dataset['dd_h'] = {'data_matrix': dd_s_matrix, 'edges': dd_s_edge_index}

    "miRNA functional sim"
    mm_f_matrix = read_csv(args.dataset_path + '/mi_mi_function.csv')
    mm_f_edge_index = get_edge_index(mm_f_matrix)
    dataset['mm_h'] = {'data_matrix': mm_f_matrix, 'edges': mm_f_edge_index}

    "RNA sequence sim"
    mm_s_matrix = read_csv(args.dataset_path + '/mi_mi_seq.csv')
    mm_s_edge_index = get_edge_index(mm_s_matrix)
    dataset['mm_s'] = {'data_matrix': mm_s_matrix, 'edges': mm_s_edge_index}

    "RNA GIP sim"
    mm_s_matrix = read_csv(args.dataset_path + '/mi_mi_GIP.csv')
    mm_s_edge_index = get_edge_index(mm_s_matrix)
    dataset['mm_g'] = {'data_matrix': mm_s_matrix, 'edges': mm_s_edge_index}

    # "RNA hamads/ sim"
    # dd_s_matrix = read_csv(args.dataset_path + '/mi_mi_hamads.csv')
    # dd_s_edge_index = get_edge_index(dd_s_matrix)
    # dataset['mm_h'] = {'data_matrix': dd_s_matrix, 'edges': dd_s_edge_index}

    "RNA_gene association"
    dataset['mi_feature'] = read_csv(args.dataset_path + '/mi_g_softmax.csv').T  #/mi_g_softmax.csv

    "disease_gene association"
    dataset['d_feature'] = read_csv(args.dataset_path + '/d_g_softmax.csv').T  #d_g_softmax

    # "miRNA_mRMA association"
    # dataset['mi_feature'] = read_csv(args.dataset_path + '/mi_m_softmax.csv').T

    # "disease_ feature'] = read_csv(args.dataset_path + '/d_m_softmax.csv').T


    return dataset

