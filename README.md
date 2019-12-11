# Graph Adversarial Learning Literature
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A curated list of adversarial attacks and defenses papers on graph-structured data.

Papers are sorted by their uploaded dates in descending order.

## Papers

### Attack
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Link        |
|-------|--------|--------|--------|-----------|------------|---------------|
| 2019 | **Time-aware Gradient Attack on Dynamic Network Link Prediction**  | Defense  | Link Prediction  | Dynamic Network Embedding Algs | Arxiv | [Link](https://arxiv.org/abs/1911.10561) |
| 2019 | **Multiscale Evolutionary Perturbation Attack on Community Detection**  | Attack  |  Community Detection   | Community Metrics | Arxiv | [Link](https://arxiv.org/abs/1910.09741) |
| 2019 | **A Unified Framework for Data Poisoning Attack to Graph-based Semi-supervised Learning**  | Attack  | Regression, Classification  | Label Propagation, Manifold Regularization | NeurIPS 2019 | [Link](https://papers.nips.cc/paper/9171-a-unified-framework-for-data-poisoning-attack-to-graph-based-semi-supervised-learning.pdf) |
| 2019 | **Attacking Graph Convolutional Networks via Rewiring**  | Attack  |  Node Classification   | GCN | ICLR 2020 Openreview | [Link](https://openreview.net/pdf?id=B1eXygBFPH) |
| 2019 | **Node Injection Attacks on Graphs via Reinforcement Learning**  | Attack  |  Node Classification    | GCN | Arxiv | [Link](https://arxiv.org/abs/1909.06543) |
| 2019 | **A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models**  | Attack  |  Node Classification   | GCN, SGC | Arxiv | [Link](https://arxiv.org/abs/1908.01297) |
| 2019 | **Unsupervised Euclidean Distance Attack on Network Embedding**  | Attack  |  Node Embedding   | GCN | Arxiv | [Link](https://arxiv.org/abs/1905.11015) |
| 2019 | **Generalizable Adversarial Attacks Using Generative Models**  | Attack  |  Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1905.10864) |
| 2019 | **Vertex Nomination, Consistent Estimation, and Adversarial Modification**  | Attack  |  Vertex Nomination   | VN Scheme | Arxiv | [Link](https://arxiv.org/abs/1905.01776) |
| 2019 | **Towards Data Poisoning Attack against Knowledge Graph Embedding**  | Attack  | Fact Plausibility Prediction   | TransE, TransR | IJCAI 2019 | [Link](https://arxiv.org/abs/1904.12052) |
| 2018 | **Adversarial Attacks on Node Embeddings via Graph Poisoning**  | Attack  | Node Classification, Community Detection   | node2vec, DeepWalk, GCN, LINE | ICML 2019| [Link](https://arxiv.org/abs/1809.01093#) |
| 2019 | **Attacking Graph-based Classification via Manipulating the Graph Structure**  | Attack  | Node Classification   | Belief Propagation, GCN | CCS 2019 | [Link](https://arxiv.org/abs/1903.00553) |
| 2019 | **Adversarial Attacks on Graph Neural Networks via Meta Learning**  | Attack  | Node Classification   | GCN, CLN, DeepWalk | ICLR 2019 | [Link](https://arxiv.org/abs/1902.08412) |
| 2018 | **GA Based Q-Attack on Community Detection**  | Attack  | Community Detection   | Modularity, Community Detection Alg | IEEE TCSS| [Link](https://ieeexplore.ieee.org/abstract/document/8714065) |
| 2018 | **Data Poisoning Attack against Unsupervised Node Embedding Methods**  | Attack  | Link Prediction   | LINE, DeepWalk | Arxiv| [Link](https://arxiv.org/abs/1810.12881) |
| 2018 | **Attack Graph Convolutional Networks by Adding Fake Nodes**  | Attack  | Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1810.10751) |
| 2018 | **Link Prediction Adversarial Attack**  | Attack  | Link Prediction   | GAE, GCN | Arxiv | [Link](https://arxiv.org/abs/1810.01110) |
| 2018 | **Attack Tolerance of Link Prediction Algorithms: How to Hide Your Relations in a Social Network**  | Attack  | Link Prediction   | Traditional Link Prediction Algs | Scientific Reports | [Link](https://arxiv.org/abs/1809.00152) |
| 2018 | **Attacking Similarity-Based Link Prediction in Social Networks**  | Attack  | Link Prediction   | local&global similarity metrics | AAMAS 2019 | [Link](https://dl.acm.org/citation.cfm?id=3306127.3331707) |
| 2018 | **Fast Gradient Attack on Network Embedding**  | Attack  | Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1809.02797) |
| 2018 | **Adversarial Attack on Graph Structured Data**  | Attack  | Node/Graph Classification   | GNN, GCN | ICML 2018 | [Link](https://arxiv.org/abs/1806.02371) |
| 2018 | **Adversarial Attacks on Neural Networks for Graph Data**  | Attack  | Node Classification   | GCN | KDD 2018 | [Link](https://arxiv.org/abs/1805.07984) |
| 2017 | **Practical Attacks Against Graph-based Clustering**  | Attack  | Graph Clustering   | SVD, node2vec, Community Detection Alg | CCS 2017| [Link](https://arxiv.org/abs/1708.09056) |

### Defense
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Link        |
|-------|--------|--------|--------|-----------|------------|---------------|
| 2019 | **GraphDefense: Towards Robust Graph Convolutional Networks**  | Defense  |  Node Classification | GCN | Arxiv | [Link](https://arxiv.org/abs/1911.04429) |
| 2019 | **All You Need is Low (Rank): Defending Against Adversarial Attacks on Graphs**  | Defense  |   |  | WSDM 2020 | Link |
| 2019 | **αCyber: Enhancing Robustness of Android Malware Detection System against Adversarial Attacks on Heterogeneous Graph based Model**  | Defense  | Malware Detection  | HIN | CIKM 2019 | [Link](https://dl.acm.org/citation.cfm?id=3357875) |
| 2019 | **Edge Dithering for Robust Adaptive Graph Convolutional Networks**  | Defense  |  Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1910.09590) |
| 2019 | **GraphSAC: Detecting anomalies in large-scale graphs**  | Defense  |  Anomaly Detection   | Anomaly Detection Algs | Arxiv | [Link](https://arxiv.org/abs/1910.09589) |
| 2019 | **Certifiable Robustness to Graph Perturbations**  | Defense  | Node Classification  | GNN | NeurIPS 2019 | [Link](https://papers.nips.cc/paper/9041-certifiable-robustness-to-graph-perturbations.pdf) |
| 2019 | **Power up! Robust Graph Convolutional Network based on Graph Powering**  | Defense  |  Node Classification   | GCN | ICLR 2020 Openreview | [Link](https://openreview.net/pdf?id=BkxDxJHFDr) |
| 2019 | **Adversarial Robustness of Similarity-Based Link Prediction**  | Defense  |  Link Prediction   | local similarity metrics | ICDM 2019 | [Link](https://arxiv.org/abs/1909.01432) |
| 2019 | **Transferring Robustness for Graph Neural Network Against Poisoning Attacks**  | Defense  |  Node Classification   | GNN | WSDM 2020 | [Link](https://arxiv.org/abs/1908.07558) |
| 2019 | **Improving Robustness to Attacks Against Vertex Classification**  | Defense  |  Node Classification   | GCN | KDD Workshop 2019 | [Link](http://eliassi.org/papers/benmiller-mlg2019.pdf) |
| 2019 | **Certifiable Robustness and Robust Training for Graph Convolutional Networks**  | Defense  |  Node Classification   | GCN | KDD 2019 | [Link](https://arxiv.org/abs/1906.12269) |
| 2019 | **Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective**  | Defense  |  Node Classification   | GNN | IJCAI 2019 | [Link](https://arxiv.org/abs/1906.04214) |
| 2019 | **Adversarial Examples on Graph Data: Deep Insights into Attack and Defense**  | Defense  | Node Classification   | GCN | IJCAI 2019 | [Link](https://arxiv.org/abs/1903.01610) |
| 2019 | **Adversarial Defense Framework for Graph Neural Network**  | Defense  | Node Classification   | GCN, GraphSAGE | Arxiv | [Link](https://arxiv.org/abs/1905.03679) |
| 2019 | **Investigating Robustness and Interpretability of Link Prediction via Adversarial Modifications**  | Defense  | Link Prediction   | Knowledge Graph Embedding | NAACL 2019 | [Link](https://arxiv.org/abs/1905.00563) |
| 2019 | **Robust Graph Convolutional Networks Against Adversarial Attacks**  | Defense  | Node Classification   | GCN | KDD 2019 | [Link](http://pengcui.thumedialab.com/papers/RGCN.pdf) |
| 2019 | **Can Adversarial Network Attack be Defended?**  | Defense  | Node Classification   | GNN | Arxiv | [Link](https://arxiv.org/abs/1903.05994) |
| 2019 | **Virtual Adversarial Training on Graph Convolutional Networks in Node Classification**  | Defense  | Node Classification   | GCN | PRCV | [Link](https://arxiv.org/abs/1902.11045) |
| 2019 | **Comparing and Detecting Adversarial Attacks for Graph Deep Learning**  | Defense | Node Classification  | GCN, GAT, Nettack | RLGM@ICLR 2019 | [Link](https://rlgm.github.io/papers/57.pdf) |
| 2019 | **Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure**  | Defense  | Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1902.08226) |
| 2018 | **Characterizing Malicious Edges targeting on Graph Neural Networks**  | Defense  | Detected Added Edges   | GNN, GCN | ICLR 2019 OpenReview | [Link](https://openreview.net/forum?id=HJxdAoCcYX) |


## Survey
**Adversarial Attack and Defense on Graph Data: A Survey** ([Link](https://arxiv.org/abs/1812.10528))

```
@article{sun2018adversarial,
  title={Adversarial Attack and Defense on Graph Data: A Survey},
  author={Sun, Lichao and Wang, Ji and Yu, Philip S and Li, Bo},
  journal={arXiv preprint arXiv:1812.10528},
  year={2018}
}
```
