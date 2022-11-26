import torch
from torch import nn
import math
from param import parameter_parser
import torch.nn.functional as F
from torch.nn.parameter import Parameter
torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha_GAT, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_GAT = alpha_GAT
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha_GAT)

    def forward(self, h, adj):
        # print("adj shape:", adj.shape)
        # print(self.W.shape)
        # print("h shape:", h.shape)
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # print("Wh shape:", Wh.shape)
        e = self._prepare_attentional_mechanism_input(Wh)
        # print("e shape:", e.shape)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha_GAT, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha_GAT=alpha_GAT, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha_GAT=alpha_GAT, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # print("x size:", x.shape)
        return x


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=True, variant=False):
        super(GraphConvolution, self).__init__()

        # nfeat, nhid, nclass, dropout, alpha_GAT, nheads
        # self.gat =  GAT(nfeat, in_features,  in_features, dropout, alpha_GAT, nheads)

        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        # print("adj size:", adj.shape)
        # print("input size:", input.shape)
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(self.weight, support) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GCNII(nn.Module):
    def __init__(self, args, alpha_GAT=0.01, nheads = 5):
        super(GCNII, self).__init__()
        self.args = args

        # nfeat: self.args.fm/fd; nlayers: self.args.layer; nhidden: self.args.hidden; dropout: self.args.dropout;
        #
        self.gat1_x = GAT(self.args.frg, self.args.fm, 512, self.args.dropout, alpha_GAT, nheads)
        self.gat2_x = GAT(512, 512, self.args.fm, self.args.dropout, alpha_GAT, 2)
        # self.gat2_x = GAT(512, 512, 256, self.args.dropout, alpha_GAT, 2)

        self.gat1_y = GAT(self.args.fdg, self.args.fd, 512, self.args.dropout, alpha_GAT, nheads)
        self.gat2_y = GAT(512, 512, self.args.fd, self.args.dropout, alpha_GAT, 2)
        # self.gat2_y = GAT(512, 512, 256, self.args.dropout, alpha_GAT, 2)

        self.convs_x = nn.ModuleList()
        for _ in range(self.args.layer):
            self.convs_x.append(GraphConvolution(self.args.fm, self.args.fm, variant=self.args.variant))
        self.fcs_x = nn.ModuleList()
        self.fcs_x.append(nn.Linear(self.args.frg, self.args.hidden))
        self.fcs_x.append(nn.Linear(self.args.hidden, self.args.fm))
        self.gat_x = nn.ModuleList()
        self.gat_x.append(self.gat1_x)
        self.gat_x.append(self.gat2_x)
        self.params1_x = list(self.convs_x.parameters())
        self.params2_x = list(self.fcs_x.parameters())
        self.params3_x = list(self.gat_x.parameters())



        self.convs_y = nn.ModuleList()
        for _ in range(self.args.layer):
            self.convs_y.append(GraphConvolution(self.args.fd, self.args.fd, variant=self.args.variant))
        self.fcs_y = nn.ModuleList()
        self.fcs_y.append(nn.Linear(self.args.fdg, self.args.hidden))
        self.fcs_y.append(nn.Linear(self.args.hidden, self.args.fd))
        self.gat_y = nn.ModuleList()
        self.gat_y.append(self.gat1_y)
        self.gat_y.append(self.gat2_y)
        self.params1_y = list(self.convs_y.parameters())
        self.params2_y = list(self.fcs_y.parameters())
        self.params3_y = list(self.gat_y.parameters())

        self.act_fn = nn.LeakyReLU(0.25)

        self.globalAvgPool_x = nn.AvgPool2d((self.args.fm, self.args.miRNA_number), (1, 1))
        self.globalAvgPool_y = nn.AvgPool2d((self.args.fd, self.args.disease_number), (1, 1))

        self.fc1_x = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
                               out_features=5 * self.args.view * self.args.gcn_layers)
        self.fc2_x = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers,
                               out_features=self.args.view * self.args.gcn_layers)

        self.fc1_y = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
                               out_features=5 * self.args.view * self.args.gcn_layers)
        self.fc2_y = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers,
                               out_features=self.args.view * self.args.gcn_layers)

        self.sigmoidx = nn.Sigmoid()
        self.sigmoidy = nn.Sigmoid()

        self.cnn_x = nn.Conv1d(in_channels=self.args.view * self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fm, 1),
                               stride=1,
                               bias=True)
        self.cnn_y = nn.Conv1d(in_channels=self.args.view * self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fd, 1),
                               stride=1,
                               bias=True)



    def forward(self, data):

        x_m = data['mi_feature'].T
        x_d = data['d_feature'].T
        # print(x_m.shape)

        # x_m = torch.randn(self.args.miRNA_number, self.args.frg)
        # x_d = torch.randn(self.args.disease_number, self.args.fdg)

        # layer_inner_x = self.act_fn(self.fcs_x[0](x_m))
        # X_M = layer_inner_x
        # layer_inner_y = self.act_fn(self.fcs_y[0](x_d))
        # Y_D = layer_inner_y

        x_m_g = self.gat1_x(x_m, data['mm_g']['data_matrix'])
        # print(x_m_g.shape)
        x_m_g = self.gat2_x(x_m_g, data['mm_g']['data_matrix'])
        # print(x_m_g.shape)
        x_m_s = self.gat1_x(x_m, data['mm_s']['data_matrix'])
        # print(x_m_s.shape)
        x_m_s = self.gat2_x(x_m_s, data['mm_s']['data_matrix'])
        # print(x_m_s.shape)
        x_m_h = self.gat1_x(x_m, data['mm_h']['data_matrix'])
        x_m_h = self.gat2_x(x_m_h, data['mm_h']['data_matrix'])

        # print(data['mm_g']['data_matrix'].shape)


        x_m = F.dropout(x_m, self.args.dropout, training=self.training)
        layer_inner_x = self.act_fn(self.fcs_x[0](x_m))
        # print(layer_inner_x.shape)

        _layers_x = []
        _layers_x.append(layer_inner_x)

        # for i, con in enumerate(self.convs_x):
            # # layer_inner_x_g = F.dropout(layer_inner_x, self.args.dropout, training=self.training)
            # layer_inner_x_g = self.act_fn(
            #     con(x_m_g, data['mm_g']['data_matrix'], _layers_x[0], self.args.lamda, self.args.alpha, i + 1))
            # # layer_inner_x_s = F.dropout(layer_inner_x, self.args.dropout, training=self.training)
            # layer_inner_x_s = self.act_fn(
            #     con(x_m_s, data['mm_s']['data_matrix'], _layers_x[0], self.args.lamda, self.args.alpha, i + 1))
            # # layer_inner_x_f = F.dropout(layer_inner_x, self.args.dropout, training=self.training)
            # layer_inner_x_f = self.act_fn(
            #     con(x_m_h, data['mm_h']['data_matrix'], _layers_x[0], self.args.lamda, self.args.alpha,
            #         i + 1))


        # for i, con in enumerate(self.convs_x):
        #     layer_inner_x_g = F.dropout(layer_inner_x, self.args.dropout, training=self.training)
        #     layer_inner_x_g = self.act_fn(
        #         con(layer_inner_x_g, data['mm_g']['data_matrix'], _layers_x[0], self.args.lamda, self.args.alpha, i + 1))
        #     layer_inner_x_s = F.dropout(layer_inner_x, self.args.dropout, training=self.training)
        #     layer_inner_x_s = self.act_fn(
        #         con(layer_inner_x_s, data['mm_s']['data_matrix'], _layers_x[0], self.args.lamda, self.args.alpha, i + 1))
        #     layer_inner_x_f = F.dropout(layer_inner_x, self.args.dropout, training=self.training)
        #     layer_inner_x_f = self.act_fn(
        #         con(layer_inner_x_f, data['mm_f']['data_matrix'], _layers_x[0], self.args.lamda, self.args.alpha,
        #             i + 1))

        for i, con in enumerate(self.convs_x):
            layer_inner_x_g = F.dropout(layer_inner_x, self.args.dropout, training=self.training)
            layer_inner_x_g = self.act_fn(
                con(layer_inner_x_g, x_m_g, _layers_x[0], self.args.lamda, self.args.alpha, i + 1))
            layer_inner_x_s = F.dropout(layer_inner_x, self.args.dropout, training=self.training)
            layer_inner_x_s = self.act_fn(
                con(layer_inner_x_s, x_m_s, _layers_x[0], self.args.lamda, self.args.alpha, i + 1))
            layer_inner_x_h = F.dropout(layer_inner_x, self.args.dropout, training=self.training)
            layer_inner_x_h = self.act_fn(
                con(layer_inner_x_h, x_m_h, _layers_x[0], self.args.lamda, self.args.alpha,
                    i + 1))


        layer_inner_x_g = F.dropout(layer_inner_x_g, self.args.dropout, training=self.training)
        layer_inner_x_g = self.fcs_x[-1](layer_inner_x_g)
        layer_inner_x_s = F.dropout(layer_inner_x_s, self.args.dropout, training=self.training)
        layer_inner_x_s = self.fcs_x[-1](layer_inner_x_s)
        layer_inner_x_h = F.dropout(layer_inner_x_h, self.args.dropout, training=self.training)
        layer_inner_x_h = self.fcs_x[-1](layer_inner_x_h)

        XM = torch.cat((layer_inner_x_g.T, layer_inner_x_s.T, layer_inner_x_h.T), 1).t()

        XM = XM.T.view(1, self.args.view * self.args.gcn_layers, self.args.fm, -1)
        x_channel_attenttion = self.globalAvgPool_x(XM)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc1_x(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc2_x(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1,
                                                         1)
        XM_channel_attention = x_channel_attenttion * XM
        XM_channel_attention = torch.relu(XM_channel_attention)


        # X_M = F.log_softmax(layer_inner_x, dim=1)
        # X_M = layer_inner_x

        x_d_g = self.gat1_y(x_d, data['dd_g']['data_matrix'])
        x_d_g = self.gat2_y(x_d_g, data['dd_g']['data_matrix'])
        x_d_s = self.gat1_y(x_d, data['dd_s']['data_matrix'])
        x_d_s = self.gat2_y(x_d_s, data['dd_s']['data_matrix'])
        x_d_h = self.gat1_y(x_d, data['dd_h']['data_matrix'])
        x_d_h = self.gat2_y(x_d_h, data['dd_h']['data_matrix'])

        x_d = F.dropout(x_d, self.args.dropout, training=self.training)
        layer_inner_y = self.act_fn(self.fcs_y[0](x_d))
        _layers_y = []
        _layers_y.append(layer_inner_y)

        # for i, con in enumerate(self.convs_y):
            # # layer_inner_y_g = F.dropout(layer_inner_y, self.args.dropout, training=self.training)
            # layer_inner_y_g = self.act_fn(
            #     con(x_d_g, data['dd_g']['data_matrix'], _layers_y[0], self.args.lamda, self.args.alpha, i + 1))
            # # layer_inner_y_s = F.dropout(layer_inner_y, self.args.dropout, training=self.training)
            # layer_inner_y_s = self.act_fn(
            #     con(x_d_s, data['dd_s']['data_matrix'], _layers_y[0], self.args.lamda, self.args.alpha, i + 1))
            # # layer_inner_y_j = F.dropout(layer_inner_y, self.args.dropout, training=self.training)
            # layer_inner_y_h = self.act_fn(
            #     con(x_d_h, data['dd_h']['data_matrix'], _layers_y[0], self.args.lamda, self.args.alpha,
            #         i + 1))

        # for i, con in enumerate(self.convs_y):
        #     layer_inner_y_g = F.dropout(layer_inner_y, self.args.dropout, training=self.training)
        #     layer_inner_y_g = self.act_fn(
        #         con(layer_inner_y_g, data['dd_g']['data_matrix'], _layers_y[0], self.args.lamda, self.args.alpha, i + 1))
        #     layer_inner_y_s = F.dropout(layer_inner_y, self.args.dropout, training=self.training)
        #     layer_inner_y_s = self.act_fn(
        #         con(layer_inner_y_s, data['dd_s']['data_matrix'], _layers_y[0], self.args.lamda, self.args.alpha, i + 1))
        #     layer_inner_y_j = F.dropout(layer_inner_y, self.args.dropout, training=self.training)
        #     layer_inner_y_j = self.act_fn(
        #         con(layer_inner_y_j, data['dd_j']['data_matrix'], _layers_y[0], self.args.lamda, self.args.alpha,
        #             i + 1))

        for i, con in enumerate(self.convs_y):
            layer_inner_y_g = F.dropout(layer_inner_y, self.args.dropout, training=self.training)
            layer_inner_y_g = self.act_fn(
                con(layer_inner_y_g, x_d_g, _layers_y[0], self.args.lamda, self.args.alpha, i + 1))
            layer_inner_y_s = F.dropout(layer_inner_y, self.args.dropout, training=self.training)
            layer_inner_y_s = self.act_fn(
                con(layer_inner_y_s, x_d_s, _layers_y[0], self.args.lamda, self.args.alpha, i + 1))
            layer_inner_y_h = F.dropout(layer_inner_y, self.args.dropout, training=self.training)
            layer_inner_y_h = self.act_fn(
                con(layer_inner_y_h, x_d_h, _layers_y[0], self.args.lamda, self.args.alpha,
                    i + 1))

        layer_inner_y_g = F.dropout(layer_inner_y_g, self.args.dropout, training=self.training)
        layer_inner_y_g = self.fcs_y[-1](layer_inner_y_g)
        layer_inner_y_s = F.dropout(layer_inner_y_s, self.args.dropout, training=self.training)
        layer_inner_y_s = self.fcs_y[-1](layer_inner_y_s)
        layer_inner_y_h = F.dropout(layer_inner_y_h, self.args.dropout, training=self.training)
        layer_inner_y_h = self.fcs_y[-1](layer_inner_y_h)

        # Y_D = layer_inner_y

        YD = torch.cat((layer_inner_y_g.T, layer_inner_y_s.T, layer_inner_y_h.T), 1).t()
        YD = YD.view(1, self.args.view * self.args.gcn_layers, self.args.fd, -1)
        y_channel_attenttion = self.globalAvgPool_y(YD)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), -1)
        y_channel_attenttion = self.fc1_y(y_channel_attenttion)
        y_channel_attenttion = torch.relu(y_channel_attenttion)
        y_channel_attenttion = self.fc2_y(y_channel_attenttion)
        y_channel_attenttion = self.sigmoidy(y_channel_attenttion)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), y_channel_attenttion.size(1), 1,
                                                         1)
        YD_channel_attention = y_channel_attenttion * YD
        YD_channel_attention = torch.relu(YD_channel_attention)

        x = self.cnn_x(XM_channel_attention)
        x = x.view(self.args.out_channels, self.args.miRNA_number).t()  # self.args.out_channels

        y = self.cnn_y(YD_channel_attention)
        y = y.view(self.args.out_channels, self.args.disease_number).t()  # self.args.out_channels

        return x.mm(y.t())


# class DMF1(nn.Module):
#     def __init__(self):
#         super(DMF1, self).__init__()
#
#         row_num = 285
#         col_num = 197
#         hidden1 = 128
#         hidden2 = 96  # 32 48 64 96
#         hidden3 = 64
#
#         self.row_model = nn.Sequential(
#             nn.Linear(col_num, hidden1),
#             nn.ReLU(),
#             # nn.LeakyReLU(0.25),
#             nn.Linear(hidden1, hidden2),
#             # nn.ReLU(),
#             # nn.Linear(hidden2, hidden3)
#         )
#
#         self.col_model = nn.Sequential(
#             nn.Linear(row_num, hidden1),
#             nn.ReLU(),
#             # nn.LeakyReLU(0.25),
#             nn.Linear(hidden1, hidden2),
#             # nn.ReLU(),
#             # nn.Linear(hidden2, hidden3)
#         )
#
#
#     def forward(self, re_matrix):
#         X = self.row_model(re_matrix)
#         # X = X + x
#         Y = self.col_model(re_matrix.t())
#         # Y = Y + y
#
#         return X.mm(Y.t())
#
# class DMF2(nn.Module):
#     def __init__(self):
#         super(DMF2, self).__init__()
#
#         row_num = 285
#         col_num = 197
#         hidden1 = 128
#         hidden2 = 96  # 32 48 64 96
#         hidden3 = 64
#
#         self.row_model = nn.Sequential(
#             nn.Linear(col_num, hidden1),
#             nn.ReLU(),
#             # nn.LeakyReLU(0.25),
#             nn.Linear(hidden1, hidden2),
#             # nn.ReLU(),
#             # nn.Linear(hidden2, hidden3)
#         )
#
#         self.col_model = nn.Sequential(
#             nn.Linear(row_num, hidden1),
#             nn.ReLU(),
#             # nn.LeakyReLU(0.25),
#             nn.Linear(hidden1, hidden2),
#             # nn.ReLU(),
#             # nn.Linear(hidden2, hidden3)
#         )
#
#
#     def forward(self, re_matrix):
#         X = self.row_model(re_matrix)
#         # X = X + x
#         Y = self.col_model(re_matrix.t())
#         # Y = Y + y
#
#         return X.mm(Y.t())
#
# class DMF3(nn.Module):
#     def __init__(self):
#         super(DMF3, self).__init__()
#
#         row_num = 285
#         col_num = 197
#         hidden1 = 128
#         hidden2 = 96  # 32 48 64 96
#         hidden3 = 64
#
#         self.row_model = nn.Sequential(
#             nn.Linear(col_num, hidden1),
#             nn.ReLU(),
#             # nn.LeakyReLU(0.25),
#             nn.Linear(hidden1, hidden2),
#             # nn.ReLU(),
#             # nn.Linear(hidden2, hidden3)
#         )
#
#         self.col_model = nn.Sequential(
#             nn.Linear(row_num, hidden1),
#             nn.ReLU(),
#             # nn.LeakyReLU(0.25),
#             nn.Linear(hidden1, hidden2),
#             # nn.ReLU(),
#             # nn.Linear(hidden2, hidden3)
#         )
#
#
#     def forward(self, re_matrix):
#         X = self.row_model(re_matrix)
#         # X = X + x
#         Y = self.col_model(re_matrix.t())
#         # Y = Y + y
#
#         return X.mm(Y.t())

class DMF(nn.Module):
    def __init__(self):
        super(DMF, self).__init__()

        row_num = 285  # mi:285; circ:515; lnc:276
        col_num = 197  # mi:197; circ:82; lnc:125
        hidden1 = 128
        hidden2 = 96  # 32 48 64 96
        hidden3 = 64

        self.row_model1 = nn.Sequential(
            nn.Linear(col_num, hidden1),
            nn.ReLU(),
            # nn.LeakyReLU(0.25),
            nn.Linear(hidden1, hidden2),
            # nn.ReLU(),
            # nn.Linear(hidden2, hidden3)
        )

        self.col_model1 = nn.Sequential(
            nn.Linear(row_num, hidden1),
            nn.ReLU(),
            # nn.LeakyReLU(0.25),
            nn.Linear(hidden1, hidden2),
            # nn.ReLU(),
            # nn.Linear(hidden2, hidden3)
        )

        self.row_model2 = nn.Sequential(
            nn.Linear(col_num, hidden1),
            nn.ReLU(),
            # nn.LeakyReLU(0.25),
            nn.Linear(hidden1, hidden2),
            # nn.ReLU(),
            # nn.Linear(hidden2, hidden3)
        )

        self.col_model2 = nn.Sequential(
            nn.Linear(row_num, hidden1),
            nn.ReLU(),
            # nn.LeakyReLU(0.25),
            nn.Linear(hidden1, hidden2),
            # nn.ReLU(),
            # nn.Linear(hidden2, hidden3)
        )

    def forward(self, intial_matrix):
        X_1 = self.row_model1(intial_matrix)
        Y_1 = self.col_model1(intial_matrix.t())
        # re_mat_1 = X_1.mm(Y_1.t())
        #
        # X_2 = self.row_model2(re_mat_1)
        # Y_2 = self.col_model2(re_mat_1.t())

        return X_1.mm(Y_1.t())


class co_CL(nn.Module):
    def __init__(self, args, alpha=0.8):
        super(co_CL, self).__init__()
        self.gcnii = GCNII(args)
        self.dmf = DMF()
        self.args = args
        self.tau = 0.5
        self.alpha = alpha
        self.fc1 = torch.nn.Linear(self.args.fm, 128)
        self.fc2 = torch.nn.Linear(128, self.args.fm)

    def forward(self, data, matrix):
        z1 = self.gcnii(data)
        h1 = self.projection(z1)
        z2_1 = self.dmf1(z1)
        z2_2 = self.dmf2(z2_1)
        z2 = self.dmf3(z2_2)
        h2 = self.projection(z2)


        loss = self.alpha*self.sim(h1, h2) + (1 - self.alpha)*self.sim(h2, h1)
        return loss

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.sum(dim=-1).mean()
        return loss

    def mix2(self, z1, z2):
        loss = ((z1 - z2) ** 2).sum() / z1.shape[0]
        return loss


def init_model_dict(args):
    model_dict = {}
    model_dict["GCNII"] = GCNII(args)
    model_dict["DMF"] = DMF()
    model_dict["co_CL"] = co_CL(args, alpha=0.8)

    return model_dict

def init_optim(model_dict, lr):
    optim_dict = {}
    optim_dict['GCNII'] = torch.optim.Adam(model_dict['GCNII'].parameters(), lr=lr)
    optim_dict['DMF'] = torch.optim.Adam(model_dict['DMF'].parameters(), lr=lr)
    optim_dict['co_CL'] = torch.optim.Adam(model_dict['co_CL'].parameters(), lr=lr)

    return optim_dict
