import argparse

parser = argparse.ArgumentParser(description='Training Config.')
parser.add_argument('--dataset', type=str, default='assistments09', help='dataset name')  # algebra05, assistments09, assistments17, nips34, statics
parser.add_argument('--logdir', type=str, default='runs/ddkc')
parser.add_argument('--savedir', type=str, default='save/ddkc/as09')
parser.add_argument('--model_name', type=str, default='ddkc', help='the model for training and evaluation')  

# TODO: Base Config
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--loss_ratio', type=float, default=0.8, help='the ratio for loss features')

# TODO: Stable Learning Config
parser.add_argument('--if_stableLearning', type=str, default="True", help='choose whether use stable learning')
parser.add_argument('--epochb', type=int, default=20, help='number of epochs to balance')
parser.add_argument('--epochp', type=int, default=0, help='number of epochs to pretrain')
parser.add_argument('--num_f', type=int, default=1, help='number of fourier spaces')
parser.add_argument('--feature_dim', type=int, default=128, help='the dim of each feature')
parser.add_argument('--lrbl', type=float, default=1.0, help='learning rate of balance')
parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
parser.add_argument('--decay_pow', type=float, default=2, help='value of pow for weight decay')
parser.add_argument('--lambdap', type=float, default=70.0, help='weight decay for weight1 ')
parser.add_argument('--lambdapre', type=float, default=1, help='weight for pre_weight1 ')
parser.add_argument('--lambda_decay_rate', type=float, default=1, help='ratio of epoch for lambda to decay')
parser.add_argument('--lambda_decay_epoch', type=int, default=5, help='number of epoch for lambda to decay')
parser.add_argument('--min_lambda_times', type=float, default=0.01, help='number of global table levels')
parser.add_argument ('--first_step_cons', type=float, default=1, help='constrain the weight at the first step')
parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')

# TODO: DCDKC_model Congfig
parser.add_argument('--eval', type=str, default="True", help='choose whether training the model')
parser.add_argument('--embed_size', type=int, default=200)
parser.add_argument('--num_attn_layers', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=8)  
parser.add_argument('--encode_pos', action='store_true')
parser.add_argument('--max_pos', type=int, default=10)
parser.add_argument('--max_length', type=int, default=200)
parser.add_argument('--grad_clip', type=float, default=10)
parser.add_argument('--hid_size', type=int, default=200)
parser.add_argument('--num_hid_layers', type=int, default=1)
args = parser.parse_args()
