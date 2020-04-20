# Graph Adversarial Learning Literature
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A curated list of adversarial attacks and defenses papers on graph-structured data.

Papers are sorted by their uploaded dates in descending order.

This **weekly-updated** list serves as a complement of the survey below.

[**Adversarial Attack and Defense on Graph Data: A Survey** ](https://arxiv.org/abs/1812.10528) **(Updated in April 2020. 35 attack papers and 30 defense papers).**

```
@article{sun2018adversarial,
  title={Adversarial Attack and Defense on Graph Data: A Survey},
  author={Lichao Sun and Yingtong Dou and Carl Yang and Ji Wang and Philip S. Yu and Bo Li},
  journal={arXiv preprint arXiv:1812.10528},
  year={2018}
}
```

If you feel this repo is helpful, please cite the survey above.


## Papers

### Attack
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Link        |  Code      |
|-------|--------|--------|--------|-----------|------------|---------------|---------|
| 2020 | **Network disruption: maximizing disagreement and polarization in social networks**  | Attack  |  Manipulating Opinion  | Graph Model, Social Network | Arxiv | [Link](https://arxiv.org/abs/2003.08377) | |
| 2020 | **Adversarial Perturbations of Opinion Dynamics in Networks**  | Attack  |  Manipulating Opinion  | Graph Model | Arxiv | [Link](https://arxiv.org/abs/2003.07010) | |
| 2020 | **Non-target-specific Node Injection Attacks on Graph Neural Networks: A Hierarchical Reinforcement Learning Approach**  | Attack  |  Node Classification  | GCN | WWW 2020 | [Link](https://faculty.ist.psu.edu/vhonavar/Papers/www20.pdf) | |
| 2020 | **MGA: Momentum Gradient Attack on Network**  | Attack  |  Node Classification, Community Detection | GCN, DeepWalk, node2vec | Arxiv | [Link](https://arxiv.org/abs/2002.11320) |     |
| 2020 | **Indirect Adversarial Attacks via Poisoning Neighbors for Graph Convolutional Networks**  | Attack  |  Node Classification | GCN | BigData 2019 | [Link](https://arxiv.org/abs/2002.08012) |    |
| 2020 | **Graph Universal Adversarial Attacks: A Few Bad Actors Ruin Graph Learning Models**  | Attack  |  Node Classification | GCN | Arxiv | [Link](https://arxiv.org/abs/2002.04784) |   [Link](https://github.com/chisam0217/Graph-Universal-Attack)  |
| 2020 | **Adversarial Attacks to Scale-Free Networks: Testing the Robustness of Physical Criteria**  | Attack  | Network Structure  | Physical Criteria | Arxiv | [Link](https://arxiv.org/abs/2002.01249) |      |
| 2020 | **Adversarial Attack on Community Detection by Hiding Individuals**  | Attack  |  Community Detection  | GCN | WWW 2020 | [Link](https://arxiv.org/abs/2001.07933) | |
| 2019 | **How Robust Are Graph Neural Networks to Structural Noise?**  | Attack  |  Node Structural Identity Prediction | GIN | Arxiv | [Link](https://arxiv.org/abs/1912.10206) |      |
| 2019 | **Time-aware Gradient Attack on Dynamic Network Link Prediction**  | Attack  | Link Prediction  | Dynamic Network Embedding Algs | Arxiv | [Link](https://arxiv.org/abs/1911.10561) |     |
| 2019 | **All You Need is Low (Rank): Defending Against Adversarial Attacks on Graphs**  | Attack  |  Node Classification | GCN, Tensor Embedding | WSDM 2020 | [Link](https://dl.acm.org/doi/abs/10.1145/3336191.3371789) | [Link](https://github.com/DSE-MSU/DeepRobust)    |
| 2019 | **αCyber: Enhancing Robustness of Android Malware Detection System against Adversarial Attacks on Heterogeneous Graph based Model**  | Attack  | Malware Detection  | HIN | CIKM 2019 | [Link](https://dl.acm.org/citation.cfm?id=3357875) |      |
| 2019 | **A Unified Framework for Data Poisoning Attack to Graph-based Semi-supervised Learning**  | Attack  |  Semi-supervised Learning   |  Label Propagation    | NeurIPS 2019 | [Link](https://arxiv.org/abs/1910.14147) |   |
| 2019 | **Manipulating Node Similarity Measures in Networks**  | Attack  | Node Similarity   | Node Similarity Measures | AAMAS 2020 | [Link](https://arxiv.org/abs/1910.11529) | |
| 2019 | **Multiscale Evolutionary Perturbation Attack on Community Detection**  | Attack  |  Community Detection   | Community Metrics | Arxiv | [Link](https://arxiv.org/abs/1910.09741) | |
| 2019 | **Attacking Graph Convolutional Networks via Rewiring**  | Attack  |  Node Classification   | GCN | Openreview | [Link](https://openreview.net/pdf?id=B1eXygBFPH) |   |
| 2019 | **Node Injection Attacks on Graphs via Reinforcement Learning**  | Attack  |  Node Classification    | GCN | Arxiv | [Link](https://arxiv.org/abs/1909.06543) |     |
| 2019 | **A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models**  | Attack  |  Node Classification   | GCN, SGC | AAAI 2020 | [Link](https://arxiv.org/abs/1908.01297) |  [Link](https://github.com/SwiftieH/GFAttack)   |
| 2019 | **Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective**  | Attack  |  Node Classification   | GNN | IJCAI 2019 | [Link](https://arxiv.org/abs/1906.04214) |   [Link](https://github.com/KaidiXu/GCN_ADV_Train)   |
| 2019 | **Unsupervised Euclidean Distance Attack on Network Embedding**  | Attack  |  Node Embedding   | GCN | Arxiv | [Link](https://arxiv.org/abs/1905.11015) |    |
| 2019 | **Generalizable Adversarial Attacks Using Generative Models**  | Attack  |  Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1905.10864) |     |
| 2019 | **Vertex Nomination, Consistent Estimation, and Adversarial Modification**  | Attack  |  Vertex Nomination   | VN Scheme | Arxiv | [Link](https://arxiv.org/abs/1905.01776) |     |
| 2019 | **Data Poisoning Attack against Knowledge Graph Embedding**  | Attack  | Fact Plausibility Prediction   | TransE, TransR | IJCAI 2019 | [Link](https://arxiv.org/abs/1904.12052) |      |
| 2019 | **Adversarial Examples on Graph Data: Deep Insights into Attack and Defense**  | Attack  | Node Classification   | GCN | IJCAI 2019 | [Link](https://arxiv.org/abs/1903.01610) |   [Link](https://github.com/DSE-MSU/DeepRobust)   |
| 2019 | **Adversarial Attacks on Node Embeddings via Graph Poisoning**  | Attack  | Node Classification, Community Detection   | node2vec, DeepWalk, GCN, Spectral Embedding, Label Propagation | ICML 2019| [Link](https://arxiv.org/abs/1809.01093#) |   [Link](https://github.com/abojchevski/node_embedding_attack)  |
| 2019 | **Attacking Graph-based Classification via Manipulating the Graph Structure**  | Attack  | Node Classification   | Belief Propagation, GCN | CCS 2019 | [Link](https://arxiv.org/abs/1903.00553) |      |
| 2019 | **Adversarial Attacks on Graph Neural Networks via Meta Learning**  | Attack  | Node Classification   | GCN, CLN, DeepWalk | ICLR 2019 | [Link](https://arxiv.org/abs/1902.08412) |  [Link](https://github.com/danielzuegner/gnn-meta-attack)  |
| 2018 | **GA Based Q-Attack on Community Detection**  | Attack  | Community Detection   | Modularity, Community Detection Alg | IEEE TCSS| [Link](https://ieeexplore.ieee.org/abstract/document/8714065) |      |
| 2018 | **Data Poisoning Attack against Unsupervised Node Embedding Methods**  | Attack  | Link Prediction   | LINE, DeepWalk | Arxiv| [Link](https://arxiv.org/abs/1810.12881) |   |
| 2018 | **Attack Graph Convolutional Networks by Adding Fake Nodes**  | Attack  | Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1810.10751) |     |
| 2018 | **Link Prediction Adversarial Attack**  | Attack  | Link Prediction   | GAE, GCN | Arxiv | [Link](https://arxiv.org/abs/1810.01110) |      |
| 2018 | **Attack Tolerance of Link Prediction Algorithms: How to Hide Your Relations in a Social Network**  | Attack  | Link Prediction   | Traditional Link Prediction Algs | Scientific Reports | [Link](https://arxiv.org/abs/1809.00152) |       |
| 2018 | **Attacking Similarity-Based Link Prediction in Social Networks**  | Attack  | Link Prediction   | local&global similarity metrics | AAMAS 2019 | [Link](https://dl.acm.org/citation.cfm?id=3306127.3331707) |    |
| 2018 | **Fast Gradient Attack on Network Embedding**  | Attack  | Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1809.02797) |     |
| 2018 | **Adversarial Attack on Graph Structured Data**  | Attack  | Node/Graph Classification   | GNN, GCN | ICML 2018 | [Link](https://arxiv.org/abs/1806.02371) |  [Link](https://github.com/Hanjun-Dai/graph_adversarial_attack)   |
| 2018 | **Adversarial Attacks on Neural Networks for Graph Data**  | Attack  | Node Classification   | GCN | KDD 2018 | [Link](https://arxiv.org/abs/1805.07984) | [Link](https://github.com/danielzuegner/nettack) |
| 2018 | **Hiding individuals and communities in a social network**  | Attack  | Community Detection   | Community Detection Algs | Nature Human Behavior | [Link](https://arxiv.org/abs/1608.00375) |     |
| 2017 | **Practical Attacks Against Graph-based Clustering**  | Attack  | Graph Clustering   | SVD, node2vec, Community Detection Alg | CCS 2017| [Link](https://arxiv.org/abs/1708.09056) |      |
| 2017 | **Adversarial Sets for Regularising Neural Link Predictors**  | Attack  | Link Prediction   | Knowledge Graph Embeddings | UAI 2017 | [Link](https://arxiv.org/abs/1707.07596) |  [Link](https://github.com/uclmr/inferbeddings)   |

### Defense
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Link        |  Code |
|-------|--------|--------|--------|-----------|------------|---------------|-------|
| 2020 | **Tensor Graph Convolutional Networks for Multi-relational and Robust Learning**  | Defense  |  Node Classification | GCN | Arxiv | [Link](https://arxiv.org/abs/2003.07729) |     |
| 2020 | **Topological Effects on Attacks Against Vertex Classification**  | Defense  |  Node Classification | GCN | Arxiv | [Link](https://arxiv.org/abs/2003.05822) |     |
| 2020 | **Towards an Efficient and General Framework of Robust Training for Graph Neural Networks**  | Defense  |  Node Classification | GCN | ICASSP 2020 | [Link](https://arxiv.org/abs/2002.10947) |    |
| 2020 | **Certified Robustness of Community Detection against Adversarial Structural Perturbation via Randomized Smoothing**  | Defense  |  Community Detection  | Community Detection Algs | WWW 2020 | [Link](https://arxiv.org/abs/2002.03421) |      |
| 2019 | **How Robust Are Graph Neural Networks to Structural Noise?**  | Defense  |  Node Structural Identity Prediction | GIN | Arxiv | [Link](https://arxiv.org/abs/1912.10206) |      |
| 2019 | **GraphDefense: Towards Robust Graph Convolutional Networks**  | Defense  |  Node Classification | GCN | Arxiv | [Link](https://arxiv.org/abs/1911.04429) |     |
| 2019 | **All You Need is Low (Rank): Defending Against Adversarial Attacks on Graphs**  | Defense  |  Node Classification | GCN, Tensor Embedding | WSDM 2020 | [Link](https://dl.acm.org/doi/abs/10.1145/3336191.3371789) | [Link](https://github.com/DSE-MSU/DeepRobust)    |
| 2019 | **αCyber: Enhancing Robustness of Android Malware Detection System against Adversarial Attacks on Heterogeneous Graph based Model**  | Defense  | Malware Detection  | HIN | CIKM 2019 | [Link](https://dl.acm.org/citation.cfm?id=3357875) |      |
| 2019 | **Edge Dithering for Robust Adaptive Graph Convolutional Networks**  | Defense  |  Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1910.09590) |        |
| 2019 | **GraphSAC: Detecting anomalies in large-scale graphs**  | Defense  |  Anomaly Detection   | Anomaly Detection Algs | Arxiv | [Link](https://arxiv.org/abs/1910.09589) |    |
| 2019 | **Certifiable Robustness to Graph Perturbations**  | Defense  | Node Classification  | GNN | NeurIPS 2019 | [Link](https://papers.nips.cc/paper/9041-certifiable-robustness-to-graph-perturbations.pdf) |   [Link](https://github.com/abojchevski/graph_cert)  |
| 2019 | **Power up! Robust Graph Convolutional Network based on Graph Powering**  | Defense  |  Node Classification   | GCN | Openreview | [Link](https://openreview.net/pdf?id=BkxDxJHFDr) |  [Link](https://www.dropbox.com/sh/p36pzx1ock2iamo/AABEr7FtM5nqwC4i9nICLIsta?dl=0)   |
| 2019 | **Adversarial Robustness of Similarity-Based Link Prediction**  | Defense  |  Link Prediction   | Local Similarity Metrics | ICDM 2019 | [Link](https://arxiv.org/abs/1909.01432) |       |
| 2019 | **Adversarial Training Methods for Network Embedding**  | Defense  |  Node Classification   | DeepWalk | WWW 2019 | [Link](https://arxiv.org/abs/1908.11514) |   [Link](https://github.com/wonniu/AdvT4NE_WWW2019)    |
| 2019 | **Transferring Robustness for Graph Neural Network Against Poisoning Attacks**  | Defense  |  Node Classification   | GNN | WSDM 2020 | [Link](https://arxiv.org/abs/1908.07558) |   [Link](https://github.com/tangxianfeng/PA-GNN)    |
| 2019 | **Improving Robustness to Attacks Against Vertex Classification**  | Defense  |  Node Classification   | GCN | KDD Workshop 2019 | [Link](http://eliassi.org/papers/benmiller-mlg2019.pdf) |       |
| 2019 | **Latent Adversarial Training of Graph Convolution Networks**  | Defense  |  Node Classification   | GCN | LRGSD@ICML | [Link](https://graphreason.github.io/papers/35.pdf) |     |
| 2019 | **Certifiable Robustness and Robust Training for Graph Convolutional Networks**  | Defense  |  Node Classification   | GCN | KDD 2019 | [Link](https://arxiv.org/abs/1906.12269) |  [Link](https://github.com/danielzuegner/robust-gcn)   |
| 2019 | **Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective**  | Defense  |  Node Classification   | GNN | IJCAI 2019 | [Link](https://arxiv.org/abs/1906.04214) |   [Link](https://github.com/KaidiXu/GCN_ADV_Train)   |
| 2019 | **Adversarial Examples on Graph Data: Deep Insights into Attack and Defense**  | Defense  | Node Classification   | GCN | IJCAI 2019 | [Link](https://arxiv.org/abs/1903.01610) |   [Link](https://github.com/DSE-MSU/DeepRobust)   |
| 2019 | **Adversarial Defense Framework for Graph Neural Network**  | Defense  | Node Classification   | GCN, GraphSAGE | Arxiv | [Link](https://arxiv.org/abs/1905.03679) |        |
| 2019 | **Investigating Robustness and Interpretability of Link Prediction via Adversarial Modifications**  | Defense  | Link Prediction   | Knowledge Graph Embedding | NAACL 2019 | [Link](https://arxiv.org/abs/1905.00563) |       |
| 2019 | **Robust Graph Convolutional Networks Against Adversarial Attacks**  | Defense  | Node Classification   | GCN | KDD 2019 | [Link](http://pengcui.thumedialab.com/papers/RGCN.pdf) |    [Link](https://github.com/DSE-MSU/DeepRobust)     |
| 2019 | **Can Adversarial Network Attack be Defended?**  | Defense  | Node Classification   | GNN | Arxiv | [Link](https://arxiv.org/abs/1903.05994) |        |
| 2019 | **Virtual Adversarial Training on Graph Convolutional Networks in Node Classification**  | Defense  | Node Classification   | GCN | PRCV 2019 | [Link](https://arxiv.org/abs/1902.11045) |        |
| 2019 | **Batch Virtual Adversarial Training for Graph Convolutional Networks**  | Defense  |  Node Classification   | GCN | LRGSD@ICML | [Link](https://arxiv.org/abs/1902.09192) |     |
| 2019 | **Comparing and Detecting Adversarial Attacks for Graph Deep Learning**  | Defense | Node Classification  | GCN, GAT, Nettack | RLGM@ICLR 2019 | [Link](https://rlgm.github.io/papers/57.pdf) |       |
| 2019 | **Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure**  | Defense  | Node Classification   | GCN | TKDE | [Link](https://arxiv.org/abs/1902.08226) |   [Link](https://github.com/fulifeng/GraphAT)   |
| 2018 | **Characterizing Malicious Edges targeting on Graph Neural Networks**  | Defense  | Detected Added Edges   | GNN, GCN |  OpenReview | [Link](https://openreview.net/forum?id=HJxdAoCcYX) |       |
| 2017 | **Adversarial Sets for Regularising Neural Link Predictors**  | Attack  | Link Prediction   | Knowledge Graph Embeddings | UAI 2017 | [Link](https://arxiv.org/abs/1707.07596) |   [Link](https://github.com/uclmr/inferbeddings)    |
