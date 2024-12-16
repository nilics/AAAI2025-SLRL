import argparse
import datetime
from component.detector import Detector
from utils import seed_all, getseedsAndtruecom, writerResToFile


def run(args):
    '''
    开始处理
    @param args: 全局参数
    '''
    seeds, com_indexs = getseedsAndtruecom(args, args.dataset)
    print("search_size, args.start", len(seeds))
    for i in range(1):
        print(f"正在处理_{args.dataset}_第{i}个节点")
        seed, com_index = seeds[i], com_indexs[i]
        detector = Detector(args, seed, com_index)
        res = detector.detect()
        writerResToFile(args, res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='twitter')
    parser.add_argument('--root', type=str, default='datasets')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_size', type=int, default=100)
    parser.add_argument('--k_ego_subG', type=int, default=3)

    # Model
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=1e-2)

    # Train
    parser.add_argument('--g_batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--search_size', type=int, default=1)
    parser.add_argument('--si', type=int, default=0.9)      # 测试相似度用
    parser.add_argument('--resfileName', type=str, default='sp_cluster')
    parser.add_argument('--ablation', type=int, default=0)
    parser.add_argument('--k', type=int, default=2)


    args = parser.parse_args()
    seed_all(args.seed)

    print('= ' * 20)
    now = datetime.datetime.now()
    print('##  Starting Time:', now.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    args.train_size = 100   # 训练集社区数量
    datasets = ['amazon', 'dblp', 'twitter', 'youtube', 'lj']
    for dataset in datasets:
        args.dataset = dataset
        run(args)

    print('## Finishing Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)