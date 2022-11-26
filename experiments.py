from matplotlib import pyplot as plt

from GAT_GCNII_DMF_CL import init_model_dict, init_optim
# from GCNIIDMF_COCL import init_model_dict, init_optim
import numpy as np
from param import parameter_parser
from dataprocessing import data_pro
import torch
import pandas as pd
from main import train_epoch
import random
import os

def seed_torch(seed=3):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

seed_torch()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

neg_pos_ratio = 1
num_epoch_pretrain = 10  #10
num_epoch = 90  #90
lr = 1e-3
file_path = "F:/PhD/Code/GCL_NcDA/data(287-197)/parameters/"

class Experiments(object):
    def __init__(self):
        super().__init__()


    def CV_triplet(self):
        args = parameter_parser()
        datasets = data_pro(args)
        train_data = datasets
        mir_dis_data = train_data['md_true']
        model_dict = init_model_dict(args)
        # print(train_data['md_p'])



        k_folds = 5  # 5, 10
        index_matrix = np.array(np.where(mir_dis_data == 1))
        positive_num = index_matrix.shape[1]
        sample_num_per_fold = int(positive_num / k_folds)

        np.random.seed(0)
        np.random.shuffle(index_matrix.T)

        metrics_tensor = np.zeros((1, 7))
        # metrics_CP = np.zeros((1, 7))
        # AUPRC = np.zeros((1,1))
        # AUC = np.zeros((1,1))
        # fpr = np.zeros((999, 1))
        # tpr = np.zeros((999, 1))
        # precision_list = np.zeros((999, 1))



        for k in range(k_folds):
            # result_kfold = np.zeros((1, 7))
            print(('CROSS VALIDATION %d' % (k + 1)).center(50, '*'))

            train_tensor = np.array(mir_dis_data, copy=True)
            if k != k_folds - 1:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold:])

            train_tensor[train_index] = 0
            # print(mir_dis_data)
            # print(train_tensor)

            print("\nPretrain GCNs...")
            optim_dict = init_optim(model_dict, lr)
            for epoch in range(num_epoch_pretrain):
                train_data['md_p'] = torch.from_numpy(train_tensor)
                train_epoch(train_data, model_dict, optim_dict, train_GCN=True, train_DMF=False)

            print("\nTraining...")
            optim_dict = init_optim(model_dict, lr)
            for epoch in range(num_epoch + 1):
                train_data['md_p'] = torch.from_numpy(train_tensor)
                predict_tensor = train_epoch(train_data, model_dict, optim_dict, train_GCN=False, train_DMF=True)

            # model = MMGCN(args)
            # model.to(device)
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            # train_data['md_p'] = torch.from_numpy(train_tensor)
            # # print(train_data['md_p'])
            # predict_tensor = train(model, train_data, optimizer, args)

            # matrix_out = predict_tensor.numpy()
            # csv_file = ['1_fold', '2_fold', '3_fold', '4_fold', '5_fold']
            # matrix_out = pd.DataFrame(predict_tensor)
            # matrix_out.to_csv(csv_file[k]+'_mi.csv')


            for i in range(10):
                metrics_tensor = metrics_tensor + self.cv_tensor_model_evaluate(mir_dis_data, predict_tensor,
                                                                                train_index, i)
                # result_kfold = result_kfold + self.cv_tensor_model_evaluate(mir_dis_data, predict_tensor,
                #                                                                 train_index, i)

        result = np.around(metrics_tensor / 50, decimals=4)  # k = 5

        # result = np.around(metrics_tensor / 100, decimals=4)  # k = 10

        return result


            # for i in range(10):
            #     AUPRC = AUPRC + self.cv_tensor_model_evaluate(mir_dis_data, predict_tensor,
            #                                                                     train_index, i)[0]
            #     AUC = AUC + self.cv_tensor_model_evaluate(mir_dis_data, predict_tensor,
            #                                                          train_index, i)[1]
            #     fpr = fpr + self.cv_tensor_model_evaluate(mir_dis_data, predict_tensor,
            #                                                          train_index, i)[2]
            #     tpr = tpr + self.cv_tensor_model_evaluate(mir_dis_data, predict_tensor,
            #                                                          train_index, i)[3]
            #     precision_list = precision_list + self.cv_tensor_model_evaluate(mir_dis_data, predict_tensor,
            #                                               train_index, i)[4]
            #     print('AUPRC:', AUPRC)
            #     print('AUC:', AUC)
            #     print('fpr:', fpr)
            #     print('tpr:', tpr)
            #     print('precision_list:', precision_list)



        #     result_k = np.around(metrics_tensor / 10, decimals=4)
        #     print('result_'+str(k)+'fold:', result_k)
        #
        # print(metrics_tensor / (k + 1))



        # AUPRC = np.around(AUPRC / 50, decimals=4)
        # AUC = np.around(AUC / 50, decimals=4)
        # fpr = np.around(fpr / 50, decimals=4)
        # tpr = np.around(tpr / 50, decimals=4)
        # precision_list = np.around(precision_list / 50, decimals=4)
        #
        # fpr_df = pd.DataFrame(fpr)
        # tpr_df = pd.DataFrame(tpr)
        # precision_list_df = pd.DataFrame(precision_list)
        # fpr_df.to_csv(file_path + 'learning_rate/0.00001_fpr.csv', header=None, columns=None)
        # tpr_df.to_csv(file_path + 'learning_rate/0.00001_tpr.csv', header=None, columns=None)
        # precision_list_df.to_csv(file_path + 'learning_rate/0.00001_precision.csv', header=None, columns=None)
        #
        #
        # return AUPRC, AUC
               # fpr, tpr, precision_list

    def LOOCV(self):
        args = parameter_parser()
        datasets = data_pro(args)
        train_data = datasets
        mir_dis_data = train_data['md_true']
        model_dict = init_model_dict(args)
        # print(train_data['md_p'])

        index_matrix = np.array(np.where(mir_dis_data == 1))
        k_folds = index_matrix.shape[1]
        positive_num = index_matrix.shape[1]
        # sample_num_per_fold = int(positive_num / k_folds)
        sample_num_per_fold = 1

        np.random.seed(0)
        np.random.shuffle(index_matrix.T)

        metrics_tensor = np.zeros((1, 7))
        # metrics_CP = np.zeros((1, 7))

        for k in range(k_folds):
            print(('LOOCV %d' % (k + 1)).center(50, '*'))

            train_tensor = np.array(mir_dis_data, copy=True)
            if k != k_folds - 1:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold:])

            train_tensor[train_index] = 0
            # print(mir_dis_data)
            # print(train_tensor)

            print("\nPretrain GCNs...")
            optim_dict = init_optim(model_dict, lr)
            for epoch in range(num_epoch_pretrain):
                train_data['md_p'] = torch.from_numpy(train_tensor)
                predict_tensor = train_epoch(train_data, model_dict, optim_dict, train_DMF=False)

            print("\nTraining...")
            optim_dict = init_optim(model_dict, lr)
            for epoch in range(num_epoch + 1):
                train_data['md_p'] = torch.from_numpy(train_tensor)
                predict_tensor = train_epoch(train_data, model_dict, optim_dict)

            for i in range(10):
                metrics_tensor = metrics_tensor + self.cv_tensor_model_evaluate(mir_dis_data, predict_tensor,
                                                                                train_index, i)

        # print(metrics_tensor / (k + 1))
        result = np.around(metrics_tensor / 50, decimals=4)

        return result


    def cv_tensor_model_evaluate(self, association_tensor, predict_tensor, train_index, seed):
        test_po_num = np.array(train_index).shape[1]
        test_index = np.array(np.where(association_tensor == 0))
        np.random.seed(seed)
        np.random.shuffle(test_index.T)
        # print(np.where((negative_index-test_index)!=0))

        test_ne_index = tuple(test_index[:, :test_po_num])
        real_score = np.column_stack(
            (np.mat(association_tensor[test_ne_index].flatten()), np.mat(association_tensor[train_index].flatten())))
        predict_score = np.column_stack(
            (np.mat(predict_tensor[test_ne_index].flatten()), np.mat(predict_tensor[train_index].flatten())))
        # real_score and predict_score are array
        return self.get_metrics(real_score, predict_score)

    def get_metrics(self, real_score, predict_score):
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[
            (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]

        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1

        TP = predict_score_matrix * real_score.T
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T

        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])


        recall_list = tpr
        precision_list = TP / (TP + FP)
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
        PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])


        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)

        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index, 0]
        accuracy = accuracy_list[max_index, 0]
        specificity = specificity_list[max_index, 0]
        recall = recall_list[max_index, 0]
        precision = precision_list[max_index, 0]

        return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]
        # return [aupr[0, 0], auc[0, 0], fpr, tpr, precision_list]


if __name__ == '__main__':

    experiment = Experiments()
    print(experiment.CV_triplet())
    # print(experiment.LOOCV())
    # print(experiment.CV_type())
