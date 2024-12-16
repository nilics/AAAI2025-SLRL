import copy

import networkx as nx
import numpy as np
import torch
import time
import random

from openpyxl import Workbook

from component.agent import Agent
from torch import optim
from grakel import Graph as gGraph
from grakel.kernels import ShortestPath
from sklearn.cluster import SpectralClustering
from component.expander import Expander
from component.graph import Graph
from utils import wr_file
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import squareform, pdist
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform



class Detector:

    def __init__(self, args, seed, com_index):
        self.args = args
        # 获取图、已知社区、种子节点
        self.graph, self.coms = self.loadDataset(args.root, args.dataset)
        self.oldKnowcoms = self.coms[-args.train_size:]   # 后100
        self.oldSeed = seed

        if args.dataset == "twitter":
            fileedge = f"datasets/{self.args.dataset}/{self.args.dataset}-1.90.ungraph.txt"
            G = self.networkx(fileedge)
            self.oldKnowcoms = self.remove_disconnected_communities(self.oldKnowcoms, G)
            print("len(communities_copy):", len(self.oldKnowcoms))

        # 获取子图（种子节点的k-ego以及已知社区的k层邻居），给所有节点重新编号，记录映射关系
        knowcomSeed_nodes = set([node for com in self.coms[-args.train_size:] for node in com] + [seed])   # 后100
        self.knowcomSeedGraph, self.old_to_new_node_mapping = self.graph.get_k_layer_subgraph_and_mapping(knowcomSeed_nodes, args.k_ego_subG)
        self.knowcomSeedGraph.setParentGraph(self.graph)
        # 反转映射以创建新节点ID映射到旧节点ID的字典
        self.new_to_old_node_mapping = {new_id: old_id for old_id, new_id in self.old_to_new_node_mapping.items()}
        self.args.old_to_new_node_mapping = self.old_to_new_node_mapping
        self.args.new_to_old_node_mapping = self.new_to_old_node_mapping

        # 给节点重新编号，获取新编号后的种子节点，已知社区
        self.knowcoms = [[self.old_to_new_node_mapping[node] for node in coms] for coms in self.oldKnowcoms]
        self.args.max_size = max(len(x) for x in self.knowcoms)
        self.train_comms = self.knowcoms
        self.seed = self.old_to_new_node_mapping[seed]
        self.com_index = com_index
        # self.computeSimiAndWrite()

        # 初始化expander
        self.device = torch.device('cuda:0')
        self.expander = self.init_expander()

    def is_connected_graph(self, graph):
        return nx.is_connected(graph)

    def remove_disconnected_communities(self, communities, G):
        connected_communities = []
        for community in communities:
            community_graph = G.subgraph(community)
            if self.is_connected_graph(community_graph):
                connected_communities.append(community)
        return connected_communities

    def networkx(self, filename):
        """--------------------------------------------------------------------------------
                     function:       把一个含有边的txt数据集表示成networkx
                     Parameters:     filename：文件名称 .txt格式
                     Returns：       G：表示成networkx的图
                    ---------------------------------------------------------------------------------"""
        fin = open(filename, 'r')
        G = nx.Graph()
        for line in fin:
            data = line.split()
            if data[0] != '#':
                G.add_edge(int(data[0]), int(data[1]))
        return G

    def loadDataset(self, root, dataset):
        '''
        加载数据集
        @param root: 根目录
        @param dataset: 数据集名称
        '''
        with open(f'{root}/{dataset}/{dataset}-1.90.ungraph.txt') as fh:
            edges = fh.read().strip().split('\n')
            edges = np.array([[int(i) for i in x.split()] for x in edges])
        with open(f'{root}/{dataset}/{dataset}-1.90.cmty.txt') as fh:
            comms = fh.read().strip().split('\n')
            comms = [[int(i) for i in x.split()] for x in comms]
        graph = Graph(edges)
        return graph, comms

    def init_expander(self):
        '''
        初始化expander
        '''
        args = self.args
        device = self.device
        expander_model = Agent(args.hidden_size).to(device)
        expander_optimizer = optim.Adam(expander_model.parameters(), lr=args.g_lr)
        expander = Expander(args, self.knowcomSeedGraph, expander_model, expander_optimizer, device,
                      max_size=args.max_size)
        return expander

    def detect(self):
        '''
        检测社区
        '''

        res = []
        pred_com = [[self.seed]]
        tic = time.time()
        for iter_num in range(2):
            if iter_num != 0:
                self.updateTraincom(pred_com[0])
            for _ in range(self.args.epochs):
                self.train_expander()
            print('=' * 50)
            print(f'迭代{iter_num}[Test]')
            if iter_num == 1:
                pred_com = [[self.seed]]
            pred_com = self.expander.generateCommunity(pred_com)
            pred_com = [x[:-1] if x[-1] == 'EOS' else x for x in pred_com]
            oldID_pred_com = [self.new_to_old_node_mapping[node] for node in pred_com[0]]
            if iter_num == 0:
                if self.args.ablation == 1:
                    # 消融实验
                    wr_file(self.oldSeed, self.com_index, oldID_pred_com, self.args)
                continue
            res = [self.oldSeed, self.com_index, oldID_pred_com]
        toc = time.time()
        print(f'Elapsed Time: {(toc - tic) // 60} min {(toc - tic) % 60}s')
        return res

    def select_lists(self, matrix, n):
        '''
        从训练集中随机选择n个社区
        @param matrix: 训练集
        @param n: batch
        @return:
        '''
        num_lists = list(range(len(matrix)))
        while len(num_lists) < n:
            num_lists = num_lists + num_lists
        random_indices = np.random.choice(num_lists, size=n, replace=True)
        selected_lists = [matrix[i] for i in random_indices]
        return selected_lists

    def train_expander(self):
        '''
        训练expander
        '''
        seeds = []
        true_coms = self.select_lists(self.train_comms, self.args.g_batch_size)
        for com in true_coms:
            seeds.append(random.choice(com))

        # Reinforcement Learning
        self.expander.trainReward(seeds, true_coms)

        # Teacher Forcing
        true_comms = random.choices(self.train_comms, k=self.args.g_batch_size)
        true_comms = [self.knowcomSeedGraph.sample_expansion_from_community(x) for x in true_comms]
        self.expander.train_from_sets(true_comms)


    def computeSimiAndWrite(self):
        communities_copy = copy.deepcopy(self.knowcoms)
        sp_graph = self.com_trans_graph(communities_copy)  # Ckv是一个二维数组，每一行代表一个已知社区（节点的标号），共10个
        sp = ShortestPath(normalize=True, with_labels=False)
        sp.fit_transform(sp_graph)
        similarity = sp.transform(sp_graph)
        simi = np.nan_to_num(similarity)
        spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed')
        labels = spectral_clustering.fit_predict(simi)
        a,b = 0, 0
        for i in range(0, len(labels)):
            if labels[i] == 0:
                a +=1
            else:
                b +=1
        print("a:",a)
        print("b:",b)

        data = simi
        workbook = Workbook()
        # 激活默认工作表
        sheet = workbook.active

        # 遍历二维列表，并将数据写入工作表
        for row_idx, row in enumerate(data):
            for col_idx, value in enumerate(row):
                sheet.cell(row=row_idx + 1, column=col_idx + 1, value=value)

        # 保存工作簿到指定文件
        workbook.save(f"AAAi/{self.args.dataset}_simi.xls")

    def updateTraincom(self, com):
        '''
        更新训练集
        @param com: 包含给定节点的局部结构
        '''
        communities_copy = copy.deepcopy(self.knowcoms)
        communities_copy.insert(0, com)
        sp_graph = self.com_trans_graph(communities_copy)  # Ckv是一个二维数组，每一行代表一个已知社区（节点的标号），共10个
        sp = ShortestPath(normalize=True, with_labels=False)
        sp.fit_transform(sp_graph)
        similarity = sp.transform(sp_graph)
        simi = np.nan_to_num(similarity)
        traincom = []
        k = self.args.k
        # traincom = self.UsingScSelectCom(simi, communities_copy, 2)
        if self.args.resfileName == "sp_cluster":
            traincom = self.UsingScSelectCom(simi, communities_copy, k)
        elif self.args.resfileName == "KMedoids":
            traincom = self.UsingKMedoidsSelectCom(simi, communities_copy, k)
        elif self.args.resfileName == "Gmm":
            traincom = self.UsingGmmSelectCom(simi, communities_copy, k)
        elif self.args.resfileName == "CengCi":
            traincom = self.UsingCengCiSelectCom(simi, communities_copy, k)
        if len(traincom) != 0:
            self.train_comms = traincom
        else:
            print("0000000")

    def com_trans_graph(self, knowcom):
        """--------------------------------------------------------------------------------
                     function:       将一组已知社区转换为最短路径形式表示的图
                     Parameters:     knowcom：给定的已知社区
                                     file_edge:网络图
                     Returns：       shortest_graph: 已知社区的最短路径图
                     ---------------------------------------------------------------------------------"""

        G = nx.from_numpy_array(self.knowcomSeedGraph.adj_mat)
        shortest_graph = []
        for com in knowcom:
            edges = G.subgraph(com).edges()
            G1 = nx.Graph()
            G1.add_nodes_from(com)
            G1.add_edges_from(edges)
            adj = np.array(nx.adjacency_matrix(G1).todense())
            shortest_graph.append(gGraph(adj))
        return shortest_graph

    def UsingScSelectCom(self, simi, communities, K=2):
        '''
        聚类，并选择包含局部结构的簇
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        '''
        # 选在其中的一类
        spectral_clustering = SpectralClustering(n_clusters=K, affinity='precomputed')
        labels = spectral_clustering.fit_predict(simi)
        # 输出聚类结果
        simis, traincom = [], []
        # print(labels)
        # a = []
        # b = []
        #
        # for i in range(0, len(labels)):
        #     if labels[i] == 0:
        #         a.append(communities[i])
        #     else:
        #         b.append(communities[i])
        # print(f"a:{a}")
        # print(f"b:{b}")
        # a = [len(line) for line in a]
        # b = [len(line) for line in b]
        # print("a:", sum(a)/len(a), len(a))
        # print("b:", sum(b) / len(b), len(b))
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                simis.append(simi[0][i])
                traincom.append(communities[i])
        return traincom


    def UsingKMedoidsSelectCom(self, simi, communities, K=2):
        '''
        聚类，并选择包含局部结构的簇
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        '''
        # 将相似性矩阵转换为距离矩阵
        distance_matrix = 1 - simi
        # 初始化K-Medoids模型
        kmedoids = KMedoids(n_clusters=K, metric='precomputed', random_state=0)

        # 训练模型
        kmedoids.fit(distance_matrix)

        # 获取聚类标签
        labels = kmedoids.labels_
        print("Cluster labels:", labels)

        # 输出聚类结果
        traincom = []
        # print(labels)
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                traincom.append(communities[i])
        return traincom


    def UsingGmmSelectCom(self, simi, communities, K=2):
        '''
        聚类，并选择包含局部结构的簇
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        '''
        # 将相似性矩阵转换为距离矩阵
        distance_matrix = squareform(pdist(simi, 'euclidean'))

        # 进行高斯混合模型聚类
        gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=0)
        gmm.fit(distance_matrix)
        labels = gmm.predict(distance_matrix)
        print("Cluster labels:", labels)

        # 输出聚类结果
        traincom = []
        # print(labels)
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                traincom.append(communities[i])
        return traincom


    def UsingCengCiSelectCom(self, simi, communities, K=2):
        '''
        聚类，并选择包含局部结构的簇
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        '''
        # 将相似性矩阵转换为距离矩阵
        # dist_matrix = squareform(pdist(simi, 'euclidean'))
        dist_matrix = 1 - simi
        # print(dist_matrix)
        np.fill_diagonal(dist_matrix, 0)

        linked = linkage(squareform(dist_matrix), 'complete')

        # 使用K指定聚类数量
        labels = fcluster(linked, K, criterion='maxclust')
        print("Cluster labels:", labels)

        # 输出聚类结果
        traincom = []
        # print(labels)
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                traincom.append(communities[i])

        return traincom


