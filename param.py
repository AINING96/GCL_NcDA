import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run MMGCN.")

    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="F:/PhD/Code/GCL_NcDA/data(287-197)",
                        help="Training datasets.")

    parser.add_argument("--pre_epoch",
                        type=int,
                        default=5,
                        help="Number of pre_training epochs. Default is 300.")

    parser.add_argument("--epoch",
                        type=int,
                        default=5,
                        help="Number of training epochs. Default is 651.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=1,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--out-channels",
                        type=int,
                        default=96,  # 48 64, 96, 128
                        help="out-channels of cnn. Default is 128.")

    parser.add_argument("--miRNA-number",
                        type=int,
                        default=285,  # miRNA:285; circRNA:515; lncRNA:276
                        help="miRNA number. Default is 285.")

    parser.add_argument("--fm",
                        type=int,
                        default=285,   # miRNA:285; circRNA:515; lncRNA:276
                        help="miRNA feature dimensions. Default is 285.")

    parser.add_argument("--disease-number",
                        type=int,
                        default=197,  # miRNA:197; circRNA:82; lncRNA:125
                        help="disease number. Default is 197.")

    parser.add_argument("--fd",
                        type=int,
                        default=197,  # miRNA:197; circRNA:82; lncRNA:125
                        help="disease number. Default is 197.")

    parser.add_argument("--frg",
                        type=int,
                        default=1789,  #miRNA:1789; circRNA:418; lncRNA:3043
                        help="gene/mRNA number. Default is 1789/1583.")

    parser.add_argument("--fdg",
                        type=int,
                        default=1789,  #miRNA:1789; circRNA:61; lncRNA:3043
                        help="gene/mRNA number. Default is 1789/1583.")


    parser.add_argument("--view",
                        type=int,
                        default=3,
                        help="views number. Default is 2(2 datasets for miRNA and disease sim)")


    parser.add_argument("--validation",
                        type=int,
                        default=4,
                        help="5 cross-validation.")

    parser.add_argument('--layer',
                        type=int,
                        default=5,  #4,5,6,7
                        help='Number of layers.')

    parser.add_argument('--hidden',
                        type=int,
                        default=256,
                        help='hidden dimensions.')

    parser.add_argument('--dropout',
                        type=float,
                        default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--alpha',
                        type=float,
                        default=0.1,
                        help='alpha_l')

    parser.add_argument('--lamda',
                        type=float,
                        default=0.5,
                        help='lamda.')

    parser.add_argument('--variant',
                        action='store_true',
                        default=False,
                        help='GCN* model.')






    return parser.parse_args()