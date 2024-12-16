import copy
from typing import Union, Optional, List, Set, Dict
import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import TruncatedSVD

import torch
from torch import nn

from .env import ExpansionEnv
from .graph import Graph
from .gnn import GraphConv
from .agent import Agent


class Expander:

    def __init__(self, args, graph: Graph, model: Agent, optimizer,
                 device: Optional[torch.device] = None,
                 max_size: int = 25,
                 k: int = 3,
                 alpha: float = 0.85,
                 gamma: float = 0.99,):
        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.n_nodes = self.graph.n_nodes
        self.max_size = max_size
        self.conv = GraphConv(graph, k, alpha)
        self.gamma = gamma
        self.args = args
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device


    def generateCommunity(self, seeds: List[list[int]], max_size: Optional[int] = None):
        '''
        生成seeds的社区
        @param seeds:社区
        @param max_size:已知社区的最大尺寸
        @return:生成的社区
        '''
        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, seeds, max_size)
        self.model.eval()
        isTrain = False
        with torch.no_grad():
            episodes, _ = self._sample_trajectories(env, isTrain)           # episodes中存放模型预测的结果，即trajectories
        return episodes

    def sample_bs_trajectories(self, seeds: List[int], max_size: Optional[int] = None):
        '''
        为seeds中结点生成轨迹
        @param seeds:bs个种子节点
        @param max_size:已知社区的最大尺寸
        @return:
        '''
        max_size = self.max_size if max_size is None else max_size
        env = ExpansionEnv(self.graph, [[s] for s in seeds], max_size)
        isTrain = True
        return self._sample_trajectories(env, isTrain)

    def eval_scores(self, pred_comm: Union[List, Set],
                     true_comm: Union[List, Set]) -> (float, float, float, float):
        """
        计算Precision, Recall, F1 and Jaccard
        @param pred_comm: 预测社区
        @param true_comm: 真实社区
        """
        intersect = set(true_comm) & set(pred_comm)
        p = len(intersect) / len(pred_comm)
        r = len(intersect) / len(true_comm)
        f = 2 * p * r / (p + r + 1e-9)
        j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
        return round(p, 4), round(r, 4), round(f, 4), round(j, 4)

    def tianchong(self, rewards, logps):
        '''
        将奖励与对数概率对齐，多余位置填0
        @param rewards: 奖励
        @param logps: 对数概率
        '''

        # 找出最大的子列表长度
        max_len = max(len(sublist) for sublist in rewards)

        # 使用numpy的pad函数在二维数组中添加填充值
        filled_rewards = np.array([np.pad(sublist, (0, max_len - len(sublist)), 'constant') for sublist in rewards])

        # 获取logps的行数和列数
        rows, cols = len(logps), len(logps[0])

        # 使用numpy的pad函数将filled_rewards的形状调整为与logps相同的形状，并在缺失的地方使用零填充
        filled_rewards = np.pad(filled_rewards, ((0, rows - len(filled_rewards)), (0, cols - filled_rewards.shape[1])),
                                'constant')
        return filled_rewards

    def trainReward(self, seeds: List[int], true_coms):
        '''
        通过奖励更新参数
        @param seeds: 一个batch的节点
        @param true_coms: 节点对应的真是社区
        '''
        bs = len(seeds)
        self.model.train()
        self.optimizer.zero_grad()
        selected_nodes, logps = self.sample_bs_trajectories(seeds)


        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        # 计算奖励
        rewards = []
        for index in range(len(selected_nodes)):
            com = selected_nodes[index]
            true_com = true_coms[index]
            r, gamma = [], 0.99
            temp_com = [com[0]]
            for node in com[1:]:
                if node != 'EOS':
                    _, _, pre_cost, _ = self.eval_scores(temp_com, true_com)
                    temp_com.append(node)
                    _, _, after_cost, _ = self.eval_scores(temp_com, true_com)
                    r.append(after_cost - pre_cost)
            reward = [np.sum(r[i] * (gamma ** np.array(range(i, len(r))))) for i in range(len(r))]
            rewards.append(reward)
        rewards = self.tianchong(rewards, logps)
        rewards = torch.from_numpy(rewards).float().to(self.device)

        mask = torch.arange(rewards.size(1), device=self.device,
                            dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()
        policy_loss = -(rewards * logps * mask).sum()
        loss = policy_loss
        loss.backward()
        self.optimizer.step()


    def train_from_sets(self, episodes: List[List[int]], max_size: Optional[int] = None):
        '''
        教师机制训练
        @param episodes:
        @param max_size:
        @return:
        '''
        max_size = self.max_size if max_size is None else max_size
        self.model.train()
        self.optimizer.zero_grad()
        env = ExpansionEnv(self.graph, [[x[0]] for x in episodes], max_size)
        bs = env.bs
        x_seeds, delta_x_nodes = env.reset()
        z_seeds = self.conv(x_seeds)
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        episode_values = [[] for _ in range(bs)]
        k = 0
        while not env.done:
            k += 1
            z_nodes += self.conv(delta_x_nodes)
            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits = self.model(*model_inputs)
            logps = []
            actions = []
            for logits, candidates, i in zip(batch_logits, batch_candidates, valid_index):
                valid_candidates = set(candidates) & (set(episodes[i]) - set(env.trajectories[i]))
                if len(valid_candidates) == 0:
                    action = len(candidates)
                else:
                    sub_idx = [idx for idx, v in enumerate(candidates) if v in valid_candidates]
                    action = sub_idx[logits[sub_idx].argmax().item()]       # 选择概率最大的节点加入社区
                actions.append(action)
                logps.append(logits[action])
            new_nodes = [x[i] if i < len(x) else 'EOS' for i, x in zip(actions, batch_candidates)]
            delta_x_nodes = env.step(new_nodes, valid_index)
            for i, v1 in zip(valid_index, logps):
                episode_logps[i].append(v1)
        # Stack and Padding
        logps = nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_logps], batch_first=True)

        lengths = torch.LongTensor([len(x) for x in env.trajectories]).to(self.device)
        mask = torch.arange(logps.size(1), device=self.device,
                            dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()
        n = mask.sum()
        policy_loss = -(1 * logps * mask).sum() / n
        policy_loss.backward()
        self.optimizer.step()

    def updateGraphAndFeatAndConv(self, z_nodes, bs, new_nodes):
        '''
        在生成社区阶段，对于新增加的节点，需要更新图与GNN中邻居矩阵
        '''
        oldId = self.args.new_to_old_node_mapping[new_nodes[0]]
        oldIdnodeKego = self.graph.parentGraph.k_ego([oldId], self.args.k_ego_subG)
        start_key = max(self.args.new_to_old_node_mapping.keys()) + 1
        newIDnode_nei = dict()
        for oldIdnode in oldIdnodeKego:
            if oldIdnode not in self.args.old_to_new_node_mapping:
                # newIDnode_nei记录：key:不在当前图中的节点, value:key节点当前图(以及新增节点)中的邻居
                self.args.old_to_new_node_mapping[oldIdnode] = start_key
                self.args.new_to_old_node_mapping[start_key] = oldIdnode
                newIDnode_nei[start_key] = set()
                start_key += 1
        # 测试
        # print("len-newIDnode_nei:", len(newIDnode_nei))
        for newIdNode in newIDnode_nei.keys():
            oldIdnode = self.args.new_to_old_node_mapping[newIdNode]
            for oldIdnei_node in self.graph.parentGraph.neighbors[oldIdnode]:
                if oldIdnei_node in self.args.old_to_new_node_mapping:
                    # 新增加的节点之间可能存在边，这样添加确保不漏掉新增加节点之间的边
                    newIDnode_nei[newIdNode].add(self.args.old_to_new_node_mapping[oldIdnode])
        if len(newIDnode_nei) != 0:
            # 存在新增加的节点，需要更新图，newIDnode_neizh
            self.graph.add_nodes_with_neighbors(newIDnode_nei)
            new_n_nodes = self.graph.n_nodes
            extended_matrix = sp.csc_matrix((new_n_nodes, bs), dtype=np.float32)
            extended_matrix[:self.n_nodes, :] = z_nodes
            z_nodes = extended_matrix
            # 更新 self.n_nodes 以反映新的节点总数
            self.n_nodes = new_n_nodes
            self.conv.updateGraph(self.graph)
        return z_nodes

    def _sample_trajectories(self, env: ExpansionEnv, isTrain):
        '''
        采样轨迹或生成社区
        @param env: 环境
        @param isTrain: 是否是训练
        @return: 选择的节点以及概率
        '''
        bs = env.bs
        # 这里x_seeds = delta_x_nodes，shape=（总结点数,bs）共bs个one-hot向量
        x_seeds, delta_x_nodes = env.reset()
        # z_seeds是经过一个图卷积的x_seeds，形状不变
        z_seeds = self.conv(x_seeds)
        # 创建一个shape=（总结点数,bs）空矩阵
        z_nodes = sp.csc_matrix((self.n_nodes, bs), dtype=np.float32)
        episode_logps = [[] for _ in range(bs)]
        new_nodes = []
        # 这里的条件是对bs个节点扩展是否结束的判断，全部结束时退出循环
        while not env.done:
            # 将每次增加的节点表示加到当前社区中节点表示上，z_nodes为当前社区所有节点的表示
            z_nodes += self.conv(delta_x_nodes)

            if isTrain == False and len(new_nodes) != 0:
                # 如果不是训练过程，每添加一个节点，更新一次图
                z_nodes = self.updateGraphAndFeatAndConv(z_nodes, bs, new_nodes)
                env.updateGraph(self.graph)
                seeds = [env.data[i][0] for i in range(bs)]
                x_seeds = env.make_single_node_encoding(seeds)
                z_seeds = self.conv(x_seeds)

            valid_index = env.valid_index
            *model_inputs, batch_candidates = self._prepare_inputs(valid_index, env.trajectories, z_nodes, z_seeds)
            batch_logits = self.model(*model_inputs)
            actions, logps = self._sample_actions(batch_logits)
            new_nodes = [x[i] if i < len(x) else 'EOS' for i, x in zip(actions, batch_candidates)]      # 这一步记录添加的节点
            # 新增加节点初始编码表示 one-hot 向量
            delta_x_nodes = env.step(new_nodes, valid_index)
            # 记录每增加一个节点对应的，对数概率、value、熵
            for i, v1 in zip(valid_index, logps):
                episode_logps[i].append(v1)

        logps = nn.utils.rnn.pad_sequence([torch.stack(x) for x in episode_logps], batch_first=True)

        # 返回最终社区env.trajectories，扩展过程中的对数概率
        return env.trajectories, logps

    def _prepare_inputs(self, valid_index: List[int], trajectories: List[List[int]],
                        z_nodes: sp.csc_matrix, z_seeds: sp.csc_matrix):
        '''
        为expander准备输入，获取动作空间、节有效点表示
        @param valid_index: 未处理完的节点
        @param trajectories: 当前社区
        @param z_nodes: 社区表示
        @param z_seeds: 种子节点表示
        @return: 降低一个维度，使用记录下标的方式
        '''
        vals_seed = []
        vals_node = []
        indptr = []
        offset = 0
        batch_candidates = []
        for i in valid_index:
            boundary_nodes = self.graph.outer_boundary(trajectories[i])     # 获取当前社区的边界节点
            candidate_nodes = list(boundary_nodes)
            # assert len(candidate_nodes)
            involved_nodes = candidate_nodes + trajectories[i]  # involved_nodes保存当前社区和其边界节点
            batch_candidates.append(candidate_nodes)  # candidates
            vals_seed.append(z_seeds.T[i, involved_nodes].todense())   # 本来是使用经过GNN，这里将其缩减维只跟involved_nodes相关的表示，可以很容易知道vals_seed中各元素长度不等
            vals_node.append(z_nodes.T[i, involved_nodes].todense())    # z_nodes是当前社区的向量表示
            indptr.append((offset, offset + len(involved_nodes), offset + len(candidate_nodes)))        # 因为各个社区/节点的向量长度不一样，这里记录
            offset += len(involved_nodes)

        vals_seed = np.array(np.concatenate(vals_seed, 1))[0]           # 将一个bs的vals_seed拼接成一个一维向量
        vals_node = np.array(np.concatenate(vals_node, 1))[0]
        vals_seed = torch.from_numpy(vals_seed).to(self.device)
        vals_node = torch.from_numpy(vals_node).to(self.device)
        indptr = np.array(indptr)                   # 由于每个节点的表示长度不一样，这里indptr用于标记start和end的位置
        # 准备输入 vals_seed记录bs种子节点的向量表示，vals_node记录当前bs个社区节点的向量表示。
        # 由于转换成一个维度，这里使用indptr区分各个节点/社区的位置
        # batch_candidates存放每个社区的候选的节点，即边界节点/动作空间
        return vals_seed, vals_node, indptr, batch_candidates

    def _sample_actions(self, batch_logits: List) -> (List, List, List):
        '''
        采样动作
        @param batch_logits: 动作空间种节点对应的对数概率
        @return:选择的节点与其对应的对数概率
        '''
        batch = []
        for logits in batch_logits:
            ps = torch.exp(logits) + 1e-8
            # 进行多项式采样
            try:
                action = torch.multinomial(ps, 1).item()
            except Exception as e:
                print(f"Caught an exception: {e}")
                print(f"ps: {ps}")
                print(f"logits: {logits}")
            logp = logits[action]
            batch.append([action, logp])
        actions, logps = zip(*batch)
        actions = np.array(actions)
        # 动作、对数概率
        return actions, logps
