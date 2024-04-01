import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes_CDC_theta
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from genotypes_CDC_theta import SEARCH_NET_1007 as genotype
import torch.nn.functional as F
import time
from torch.autograd import Variable
from model import NetworkCIFAR as Network
from my_dataset import My_Dataset

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=24, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=12, help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='./indataset_pth_txt/eval-EXP-cross_domain_train_Celeb-DF2/pth/138.pt',
                    help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='SEARCH_NET_1007', help='which architecture to use')
parser.add_argument('--method', type=str, default='CrossDomain')
parser.add_argument('--testMethod', type=str, default='Cross-Dataset')
parser.add_argument('--dataset', type=str, default='Celebdf', help='gradient clipping')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 2
save_pth = 'test-{}-{}-{}-{}'.format(args.testMethod, args.method, args.dataset, time.strftime("%Y%m%d"))
utils.create_exp_dir(save_pth, scripts_to_save=glob.glob('*.py'))
fh = logging.FileHandler(os.path.join(save_pth, f'{args.dataset}_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes_CDC_theta.%s" % args.arch)
    # print('genotype:', genotype)
    logging.info('genotype:%s', genotype)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()
    print('model_path:', args.model_path)
    utils.load(model, args.model_path)
    # # layer_name = list(model.state_dict().keys())
    # # # print('look:',model.state_dict()[layer_name[2]])
    logging.info("test dataset:%s", args.dataset)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    _, test_transform = utils._data_transforms_CASIAv2()

    # test_file = '/media/ubuntu/hdd/PycharmProjects/XXX/PC-DARTS-DeepFake/face_data_process/Deepfake_data_process/npy/002/test/Deepfakes_test/Deepfakes_test_data.npy'
    # test_num = 5600
    test_file = '/media/ubuntu/hdd/PycharmProjects/XXX/PC-DARTS-DeepFake-theta/face_data_process/Celeb-DF2_data_process/npy/002/Celeb_DF2_test_data_18_10frames.npy'
    test_num = 6081
    # test_file = '/media/ubuntu/hdd/PycharmProjects/XXX/PC-DARTS-DeepFake-theta/face_data_process/DFDC_preview_process/npy/002/DFDC_Pre_test_data_20_10frames.npy'
    # test_num = 8347
    # test_file = '/media/ubuntu/hdd/PycharmProjects/XXX/PC-DARTS-DeepFake-theta/face_data_process/WildDeepfake_data_process/npy/001/WildDeepfake_test_data_10frames.npy'
    # test_num = 8060

    test_data = My_Dataset(test_num, test_file, choice='train', transform=test_transform)
    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    print(test_num, test_file)
    model.drop_path_prob = args.drop_path_prob
    test_acc, test_obj, test_auc = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)
    logging.info('test_auc %f', test_auc)


def infer(test_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    test_auc = utils.AvgrageMeter()
    # test_auc_list = []
    # test_f1_list = []
    # auc_list, f1_list = [], []
    model.eval()

    for step, (input, target) in enumerate(test_queue):
        input = input.to('cuda')
        target = target.to('cuda')

        logits, _ = model(input)
        # print(logits.shape) # torch.Size([24, 2])
        loss = criterion(logits, target)
        prob = F.softmax(logits, dim=1).data.cpu().numpy()[:, 1]

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 1))
        auc = utils.AUC(prob, target.data.cpu().numpy())
        test_auc.update(auc, logits.shape[0])
        # f1 = utils.f1_score(utils.prob_to_cls(prob, threshold=0.5), target.data.cpu().numpy())
        # auc_list.append(auc)
        # f1_list.append(f1)

        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    # test_auc_list.append(np.mean(auc_list))
    # test_f1_list.append(np.mean(f1_list))
    # return top1.avg, objs.avg, test_auc_list[-1], test_f1_list[-1]
    return top1.avg, objs.avg, test_auc.avg


if __name__ == '__main__':
    main()
