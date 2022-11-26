
from GAT_GCNII_DMF_CL import init_model_dict, init_optim
# from GCNIIDMF_COCL import init_model_dict, init_optim
from param import parameter_parser
from dataprocessing import data_pro
import torch.nn.functional as F
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(data,model_dict,optim_dict, train_GCN = True, train_DMF = True):
    loss_dict = {}
    loss = torch.nn.MSELoss(reduction='mean')
    if train_GCN:
        for m in model_dict:
            model_dict[m].train()

        optim_dict['GCNII'].zero_grad()
        score = model_dict["GCNII"](data)
        loss = loss(score, data['md_p'].to(device))
        print("GCNII loss:", loss.item())
        loss.backward()
        optim_dict['GCNII'].step()
        loss_dict['GCNII'] = loss.detach().cpu().numpy().item()


    if train_DMF:
        # x = model_dict["GCN"](data)[0]
        # y = model_dict["GCN"](data)[1]
        optim_dict["DMF"].zero_grad()
        score = model_dict["GCNII"](data)
        loss_GCN = loss(score, data['md_p'].to(device)).item()
        for i in range(score.shape[0]):
            for j in range(score.shape[1]):
                if data['md_true'][i][j] == 1:
                    score[i][j] = 1
        pre_score1 = model_dict["DMF"](score)
        new_loss1 = torch.nn.SmoothL1Loss()   # new_loss = torch.nn.MSELoss(reduction='mean')


        alpha = 0.8
        z1 = score
        h1 = projection(z1)
        z2_1 = pre_score1
        h2 = projection(z2_1)
        co_loss1 = (alpha * sim(h1, h2) + (1 - alpha) * sim(h2, h1))/torch.tensor([197*285])
        print("co_loss:", co_loss1.item())

        DMF_loss1 = new_loss1(pre_score1, data['md_p'].to(device))
        # pre_loss1 = DMF_loss1
        pre_loss1 = (DMF_loss1 + co_loss1 + loss_GCN)/3

        # pre_loss1 = new_loss1(pre_score1, data['md_p'].to(device))
        pre_loss1.backward()

        optim_dict["DMF"].step()
        loss_dict['DMF'] = DMF_loss1.item()
        print("DMF_loss:", pre_loss1.item())
        # print("pre_loss:", pre_loss1.item())

        # score1 = model_dict["DMF1"](pre_score1)
        # optim_dict["DMF2"].zero_grad()
        # for i in range(score.shape[0]):
        #     for j in range(score.shape[1]):
        #         if data['md_true'][i][j] == 1:
        #             pre_score1[i][j] = 1
        # pre_score2 = model_dict["DMF2"](pre_score1)
        # new_loss2 = torch.nn.SmoothL1Loss()   # new_loss = torch.nn.MSELoss(reduction='mean')
        #
        # alpha = 0.8
        # z1 = score
        # h1 = projection(z1)
        # z2 = pre_score1
        # h2 = projection(z2)
        # co_loss2 = (alpha * sim(h1, h2) + (1 - alpha) * sim(h1, h2))/torch.tensor([197*285])
        # print("co_loss:", co_loss2.item())
        #
        # DMF_loss2 = new_loss2(pre_score2, data['md_p'].to(device))
        # pre_loss2 = DMF_loss2 + loss_GCN + co_loss2
        # # torch.autograd.set_detect_anomaly(True)
        # new_pre_loss2 = pre_loss2.detach_().requires_grad_(True)
        # new_pre_loss2.backward()
        # optim_dict["DMF2"].step()
        # loss_dict['DMF2'] = DMF_loss2.item()
        # print("DMF2_loss:", DMF_loss2.item())
        # print("pre_loss:", pre_loss2.item())
        #
        # optim_dict["DMF3"].zero_grad()
        # for i in range(score.shape[0]):
        #     for j in range(score.shape[1]):
        #         if data['md_true'][i][j] == 1:
        #             pre_score2[i][j] = 1
        # pre_score3 = model_dict["DMF3"](pre_score2)
        # new_loss3 = torch.nn.SmoothL1Loss()   # new_loss = torch.nn.MSELoss(reduction='mean')
        #
        # alpha = 0.8
        # z1 = score
        # z2_3 = pre_score3
        # co_loss3 = alpha * sim(z1, z2_3) + (1 - alpha) * sim(z2_3, z1)
        #
        # pre_loss3 = new_loss3(pre_score3, data['md_p'].to(device)) + co_loss3
        # new_pre_loss3 = pre_loss3.detach_().requires_grad_(True)
        # new_pre_loss3.backward()
        # optim_dict["DMF3"].step()
        # loss_dict['DMF3'] = pre_loss3.item()
        # print("DMF3_loss:", pre_loss3.item())

        predict_score = pre_score1.detach().cpu().numpy()
        scoremin, scoremax = predict_score.min(), predict_score.max()
        predict_score = (predict_score - scoremin) / (scoremax - scoremin)
        return predict_score
    else:
        predict_score = score.detach().cpu().numpy()
        scoremin, scoremax = predict_score.min(), predict_score.max()
        predict_score = (predict_score - scoremin) / (scoremax - scoremin)
        return predict_score

    return loss_dict


def projection(z):
    fc1 = torch.nn.Linear(197, 128)  # 197; 82; 125
    fc2 = torch.nn.Linear(128, 197)
    z = F.elu(fc1(z))
    return fc2(z)


def norm_sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def sim(z1, z2):
    tau = 0.5
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(norm_sim(z1, z1))
    between_sim = f(norm_sim(z1, z2))
    loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    loss = loss.sum(dim=-1).mean()
    return loss


def main(num_epoch_pretrain,num_epoch, lr):
    args = parameter_parser()
    datasets = data_pro(args)
    train_data = datasets
    metrics_tensor = np.zeros((1, 7))
    k = 1

    model_dict = init_model_dict(args)
    # print(model_dict)

    # for k in range(k_folds):
    #     print(('CROSS VALIDATION %d' % (k + 1)).center(50, '*'))
    #     print("\nPretrain GCNs...")
    #     optim_dict = init_optim(model_dict, lr)
    #     for epoch in range(num_epoch_pretrain):
    #         train_epoch(train_data, model_dict, optim_dict, k, train_DMF=False)
    #     print("\nTraining...")
    #     optim_dict = init_optim(model_dict, lr)
    #     for epoch in range(num_epoch + 1):
    #         predict_tensor = train_epoch(train_data, model_dict, optim_dict, k)
    #     # print(predict_tensor)
    #     for i in range(10):
    #         metrics_tensor = metrics_tensor + cv_tensor_model_evaluate(datasets['md_true'], predict_tensor,
    #                                                                    datasets['CV_index']['CV{:}'.format(k+1)], k)
    #     # print(metrics_tensor)
    # result = np.around(metrics_tensor / 50, decimals=4)
    # return result


    print("\nPretrain GATGCNIIs...")
    optim_dict = init_optim(model_dict,lr)
    for epoch in range(num_epoch_pretrain):
        train_epoch(train_data,model_dict,optim_dict,train_DMF=False)

    print("\nTraining...")
    optim_dict = init_optim(model_dict, lr)
    for epoch in range(num_epoch+1):
         predict_tensor = train_epoch(train_data,model_dict, optim_dict)
    # print(predict_tensor)
    for i in range(10):
        metrics_tensor = metrics_tensor + cv_tensor_model_evaluate(datasets['md_true'], predict_tensor,
                                                                        datasets['CV_index'], k)

    result = np.around(metrics_tensor / 50, decimals=4)
    return result

def cv_tensor_model_evaluate(association_tensor, predict_tensor, train_index, seed):
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
    return get_metrics(real_score, predict_score)

def get_metrics(real_score, predict_score):
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

