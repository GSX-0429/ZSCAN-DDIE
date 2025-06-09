import os.path as osp
import argparse
import os
import torch
import time
from utils.Configuration import Config
from utils.Tools import mkdir_or_exist, set_random_seed
from train import train_model, evaluate
from utils.Logging_ import get_root_logger
from models.builder import build_classifier
from datasets.builder import build_dataset

CUDA_LAUNCH_BLOCKING = 1


def parse_args():
    parser = argparse.ArgumentParser(description='Train a ZSCAN=DDIE')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')

    parser.add_argument(
        '--device', default="cuda:0", help='cuda:0 or cuda:1')
    parser.add_argument(
        '--seednumber', default=42, help='number of seeds')

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--deterministic',
        default=True,
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
    #parser.add_argument('--test', type=str, default='no', help='zsl or gzsl or no')
    parser.add_argument('--zsl_para', type=str, default=False)
    parser.add_argument('--gzsl_para', type=str, default=False)

    args = parser.parse_args()

    # add args.local_rank into os.environ
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def unify_seed_device(cfg, i, det, dev,args):
    """
    Unify all seeds and devices in a work.
    """
    set_random_seed(i, deterministic=det)

    cfg.device = dev
    cfg.model.device = dev
    cfg.model.structuralmodule.device = dev
    cfg.model.textualmodule.device = dev
    cfg.data.train.device = dev
    #if not args.case:
    cfg.data.zsl_test.device = dev
    cfg.data.zsl_val.device = dev
    cfg.data.gzsl_test.device = dev
    cfg.data.gzsl_val.device = dev
    cfg.data.val_seen.device = dev
    cfg.data.test_seen.device = dev


def main():
    # 解析命令行参数
    args = parse_args()
    # 从文件中加载配
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 根据命令行参数设置工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # 设置恢复检查点
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    # 设置 GPU ID
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # 创建工作目录
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    mkdir_or_exist(osp.join(cfg.work_dir, 'model_parameter'))
    cfg.model_parameter_epoch = osp.join(osp.join(cfg.work_dir, 'model_parameter'),
                                         f'model_epoch{cfg.num_epochs}_seed{args.seednumber}.pkl')
    cfg.model_parameter_best = osp.join(osp.join(cfg.work_dir, 'model_parameter'),
                                        f'model_best_epoch{cfg.num_epochs}_seen{args.seednumber}.pkl')
    # 获取当前时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.zsl_para or args.gzsl_para:
        log_file = osp.join(cfg.work_dir, f'eval_{timestamp}.log')
    else:
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # 初始化日志记录器
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # logger.info(f"config is {cfg}")
    #logger.info(f'Set random seed to {int(args.seednumber)}')

    # 设置随机种子和设备
    # pytorch has seed, cuda has seed
    unify_seed_device(cfg, int(args.seednumber), args.deterministic, cfg.device,args)
    cfg.seednumber = int(args.seednumber)

    # 判断是进行评估还是训练
    if args.zsl_para or args.gzsl_para:  # 评估模式
        # 构建训练数据集
        train_dataset = build_dataset(cfg.data.train)
        cfg.model.textualmodule.input_dim = train_dataset.input_dim
        cfg.model.seen_labels = train_dataset.current_dataset_eventid_uni
        # 构建零样本测试数据
        zsl_test_dataset = build_dataset(cfg.data.zsl_test)
        # 构建广义零样本测试数据集
        gzsl_test_dataset = build_dataset(cfg.data.gzsl_test)
        # 构建已见测试数据集
        seen_test_dataset = build_dataset(cfg.data.test_seen)

        # 设置模型输入和输出维度
        cfg.model.textualmodule.output_dim = zsl_test_dataset.dim
        cfg.model.zsl_labels = zsl_test_dataset.current_dataset_eventid_uni
        cfg.model.gzsl_labels = gzsl_test_dataset.current_dataset_eventid_uni

        # 设置模型输入数据
        cfg.model.train_textualinput = seen_test_dataset.textualinput
        cfg.model.val_zsl_textualinput = zsl_test_dataset.textualinput
        cfg.model.val_gzsl_textualinput = gzsl_test_dataset.textualinput

        # 构建分类器模型
        model = build_classifier(cfg.model)
        model2 = build_classifier(cfg.model)
        # 将模型移动到指定设备
        model.to(cfg.device)
        model2.to(cfg.device)

        # 加载预训练模型
        # print(args.zsl_para)
        #model.load_state_dict(torch.load(args.zsl_para))
        #model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.zsl_para,map_location=cfg.device).items()})
        model.load_state_dict(torch.load(args.zsl_para))
        # print(args.gzsl_para)
        model2.load_state_dict(torch.load(args.gzsl_para))

        # seen_labels = test_dataset.seen_labels
        # 评估模型
        acc = evaluate(model, zsl_test_dataset, logger, cfg, "zsl", visualize_acc=True)
        H = evaluate(model2, gzsl_test_dataset, logger, cfg, "gzsl", visualize_acc=True)

    else:  # 训练模式
        print("train")
        datasets = []
        # 构建训练数据集
        train_dataset = build_dataset(cfg.data.train)
        cfg.model.textualmodule.input_dim = train_dataset.input_dim
        cfg.model.textualmodule.output_dim = train_dataset.dim
        cfg.model.seen_labels = train_dataset.current_dataset_eventid_uni
        # 构建零样本验证数据集
        zsl_val_dataset = build_dataset(cfg.data.zsl_val)
        # 构建广义零样本验证数据集
        gzsl_val_dataset = build_dataset(cfg.data.gzsl_val)
        cfg.model.zsl_labels = zsl_val_dataset.current_dataset_eventid_uni
        cfg.model.gzsl_labels = gzsl_val_dataset.current_dataset_eventid_uni
        # 添加数据集到列表
        datasets.append(train_dataset)
        datasets.append(zsl_val_dataset)
        datasets.append(gzsl_val_dataset)
        # 设置模型输入数据
        cfg.model.train_textualinput = train_dataset.textualinput
        cfg.model.val_zsl_textualinput = zsl_val_dataset.textualinput
        cfg.model.val_gzsl_textualinput = gzsl_val_dataset.textualinput
        cfg.model.attributlabel = train_dataset.textualattributelabel
        # 构建分类器模型
        model = build_classifier(cfg.model)
        # print("cfg.device",cfg.device)
        model.to(cfg.device)
        logger.info(f"model is {model}")

        # 开始训练模型
        print("begin to train model")
        train_model(model, datasets, cfg)


if __name__ == "__main__":
    main()