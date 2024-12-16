from typing import Union, Optional, List, Set, Dict
import collections

import numpy as np
from scipy import sparse as sp
import random


class Graph:

    def __init__(self, edges):
        self.neighbors, self.n_nodes, self.adj_mat, self.degree = self._init_from_edges(edges)

    @staticmethod
    def _init_from_edges(edges: np.ndarray) -> (Dict[int, Set[int]], int, sp.spmatrix):
        neighbors = collections.defaultdict(set)
        degrees = {}
        max_id = -1
        for u, v in edges:
            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1

            max_id = max(max_id, u, v)
            if u != v:
                neighbors[u].add(v)
                neighbors[v].add(u)
        n_nodes = len(neighbors)
        if (max_id + 1) != n_nodes:
            raise ValueError('Please re-label nodes first!')
        adj_mat = sp.csr_matrix((np.ones(len(edges)), edges.T), shape=(n_nodes, n_nodes))
        adj_mat += adj_mat.T
        return neighbors, n_nodes, adj_mat, degrees

    def setParentGraph(self, parentGraph):
        self.parentGraph = parentGraph

    def outer_boundary(self, nodes: Union[List, Set]) -> Set[int]:
        '''
        获取节点集的边界
        @param nodes:
        @return:
        '''
        boundary = set()
        for u in nodes:
            boundary |= self.neighbors[u]
        boundary.difference_update(nodes)
        return boundary

    def k_ego(self, nodes: Union[List, Set], k: int) -> Set[int]:
        '''
        获取kego网络
        @param nodes: 节点集
        @param k: k
        @return:
        '''
        ego_nodes = set(nodes)
        current_boundary = set(nodes)
        for _ in range(k):
            current_boundary = self.outer_boundary(current_boundary) - ego_nodes
            ego_nodes |= current_boundary
        return ego_nodes

    def get_k_layer_subgraph_and_mapping(self, node_list: Union[List[int], Set[int]], k: int):
        '''
        获取节点集合的kego子图，给节点重新编号，返回映射结果
        @param node_list: 节点集合
        @param k:
        '''

        # 获取k层邻居节点集
        k_layer_neighbors = self.k_ego(node_list, k)

        # 创建节点重新映射，旧节点ID映射到新节点ID
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(k_layer_neighbors))}

        old_to_new_node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(k_layer_neighbors))}

        # 创建一个边的列表来存储子图的边，使用新的编号
        subgraph_edges = []

        # 遍历k层邻居节点集，获取边
        for node in k_layer_neighbors:
            for neighbor in self.neighbors[node]:
                if neighbor in k_layer_neighbors:
                    # 使用新的节点编号
                    mapped_node = node_mapping[node]
                    mapped_neighbor = node_mapping[neighbor]
                    # 确保每条边只添加一次
                    if mapped_node < mapped_neighbor:
                        subgraph_edges.append((mapped_node, mapped_neighbor))

        # 转换边列表为NumPy数组
        edges_array = np.array(subgraph_edges)

        # 创建一个新的Graph实例作为子图
        subgraph = Graph(edges_array)

        return subgraph, node_mapping

    def sample_expansion_from_community(self, comm_nodes: Union[List, Set],
                                        seed: Optional[int] = None) -> List[int]:
        '''
        教师机制采样：随机从一个节点开始，生成真实社区，walk记录新顺序的真实社区
        @param comm_nodes:
        @param seed:
        @return:
        '''
        if seed is None:
            seed = random.choice(tuple(comm_nodes))

        remaining = set(comm_nodes) - {seed}
        boundary = self.neighbors[seed].copy()
        #         print("len(remaining):",len(remaining));
        #         print("len(boundary):",len(boundary));
        walk = [seed]
        while len(remaining):
            try:
                candidates = tuple(boundary & remaining)
                new_node = random.choice(candidates)
                remaining.remove(new_node)
                boundary |= self.neighbors[new_node]
                walk.append(new_node)
            except Exception:
                return walk
        # 随机从一个节点开始，生成真实社区，walk记录新顺序的真实社区
        return walk


    def add_nodes_with_neighbors(self, newIDnode_nei: Dict[int, Union[List[int], Set[int]]]):
        '''
        扩展种子节点所在社区时，动态更新图
        @param newIDnode_nei: 新增节点
        '''
        for new_node, neighbors in newIDnode_nei.items():
            if new_node not in self.neighbors:  # 如果新节点还不在图中
                self.n_nodes += 1  # 更新节点计数
                self.neighbors[new_node] = set()  # 初始化新节点的邻居集合

            for neighbor in neighbors:
                self.neighbors[new_node].add(neighbor)  # 添加新节点的邻居
                if neighbor not in self.neighbors:  # 如果邻居节点也是新的
                    self.n_nodes += 1  # 更新节点计数
                    self.neighbors[neighbor] = {new_node}  # 初始化邻居节点的邻居集合
                else:
                    self.neighbors[neighbor].add(new_node)  # 更新现有邻居的邻居集合

                # 更新度
                self.degree[new_node] = self.degree.get(new_node, 0) + 1
                self.degree[neighbor] = self.degree.get(neighbor, 0) + 1

        # 重新构建邻接矩阵
        edges_list = []
        for node, nbs in self.neighbors.items():
            for nb in nbs:
                if node < nb:  # 确保每条边只添加一次
                    edges_list.append((node, nb))

        edges_array = np.array(edges_list)
        self.adj_mat = sp.csr_matrix((np.ones(len(edges_list)), edges_array.T), shape=(self.n_nodes, self.n_nodes))
        self.adj_mat += self.adj_mat.T  # 使邻接矩阵对称

