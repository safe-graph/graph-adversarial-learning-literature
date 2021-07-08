# Graph Adversarial Learning Literature
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

A curated list of adversarial attacks and defenses papers on graph-structured data.

Papers are sorted by their uploaded dates in descending order.

This **bi-weekly-updated** list serves as a complement of the survey below.

[**Adversarial Attack and Defense on Graph Data: A Survey** ](https://arxiv.org/abs/1812.10528) **(Updated in July 2020. More than 110 papers reviewed).**

```bibtex
@article{sun2018adversarial,
  title={Adversarial Attack and Defense on Graph Data: A Survey},
  author={Sun, Lichao and Dou, Yingtong and Yang, Carl and Wang, Ji and Yu, Philip S. and He, Lifang and Li, Bo},
  journal={arXiv preprint arXiv:1812.10528},
  year={2018}
}
```

If you feel this repo is helpful, please cite the survey above.

## How to Search?
Search keywords like conference name (e.g., ```NeurIPS```), task name (e.g., ```Link Prediction```), model name (e.g., ```DeepWalk```), or method name (e.g., ```Robust```) over the webpage to quickly locate related papers.

## Quick Links
**Attack papers sorted by year:** | [2021](#attack-papers-2021) | [2020](#attack-papers-2020-back-to-top) | [2019](#attack-papers-2019-back-to-top) | [2018](#attack-papers-2018-back-to-top) | [2017](#attack-papers-2017-back-to-top) |

**Defense papers sorted by year:** | [2021](#defense-papers-2021-back-to-top) | [2020](#defense-papers-2020-back-to-top) | [2019](#defense-papers-2019-back-to-top) | [2018](#defense-papers-2018-back-to-top) |

## Attack

### Attack Papers 2021
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Paper        |  Code      |
|-------|--------|--------|--------|-----------|------------|---------------|---------|
| 2021 | **Adversarial Attack on Graph Neural Networks as An Influence Maximization Problem**  | Attack  |   Node Classification   | GNNs | Arxiv | [Link](https://arxiv.org/abs/2106.10785) | |
| 2021 | **BinarizedAttack: Structural Poisoning Attacks to Graph-based Anomaly Detection**  | Attack  |   Anomaly Detection   | Graph Anomaly Detection Algs | Arxiv | [Link](https://arxiv.org/abs/2106.09989) | |
| 2021 | **TDGIA: Effective Injection Attacks on Graph Neural Networks**  | Attack  |  Node Classification   | GNNs | KDD 2021 | [Link](https://arxiv.org/abs/2106.06663) |  |
| 2021 | **Graph Adversarial Attack via Rewiring**  | Attack  |  Node Classification   | GCN | KDD 2021 | [Link](https://arxiv.org/abs/1906.03750) |   |
| 2021 | **Evaluating Graph Vulnerability and Robustness using TIGER**  | Attack  |  Robustness Measure   | Robustness Measure | Arxiv | [Link](https://arxiv.org/abs/2006.05648) | [Link](https://github.com/safreita1/TIGER) |
| 2021 | **Adversarial Attack Framework on Graph Embedding Models with Limited Knowledge**  | Attack  |  Node Classification   | Graph Embedding Models | Arxiv | [Link](https://arxiv.org/abs/2105.12419) |  |
| 2021 | **Attacking Graph Neural Networks at Scale**  | Attack  |  Node Classification   | GCN | AAAI 2021 Workshop | [Link](https://www.dropbox.com/s/ddrwoswpz3wwx40/Robust_GNNs_at_Scale__AAAI_Workshop_2020_CameraReady.pdf?dl=0) |  |
| 2021 | **Black-box Gradient Attack on Graph Neural Networks: Deeper Insights in Graph-based Attack and Defense**  | Attack  |  Node Classification   | GNNs | Arxiv | [Link](https://arxiv.org/abs/2104.15061) |  |
| 2021 | **Enhancing Robustness and Resilience of Multiplex Networks Against Node-Community Cascading Failures**  | Attack  |  Complex Networks Robustness | Complex Networks | IEEE TSMC | [Link](https://ieeexplore.ieee.org/abstract/document/9415463/authors#authors) |  |
| 2021 | **PATHATTACK: Attacking Shortest Paths in Complex Networks**  | Attack  |  Shortest Path  |  Shortest Path | Arxiv | [Link](https://arxiv.org/abs/2104.03761) |  |
| 2021 | **Universal Spectral Adversarial Attacks for Deformable Shapes**  | Attack  |  Shape Classification  | ChebyNet, PointNet | CVPR 2021 | [Link](https://arxiv.org/abs/2104.03356) |  |
| 2021 | **Preserve, Promote, or Attack? GNN Explanation via Topology Perturbation**  | Attack  |  Object Detection  | GNNs | Arxiv | [Link](https://arxiv.org/abs/2103.13944) |  |
| 2021 | **Towards Revealing Parallel Adversarial Attack on Politician Socialnet of Graph Structure**  | Attack  | Node Classification  | GCN | Security and Communication Networks | [Link](https://www.hindawi.com/journals/scn/2021/6631247/) |  |
| 2021 | **Network Embedding Attack: An Euclidean Distance Based Method**  | Attack  | Node Classification, Community Detection  | Network Embedding Methods | MDATA | [Link](https://link.springer.com/chapter/10.1007%2F978-3-030-71590-8_8) |  |
| 2021 | **Adversarial Attack on Network Embeddings via Supervised Network Poisoning**  | Attack  |  Node Classification, Link Prediction  | DeepWalk, Node2vec, LINE, GCN | PAKDD 2021 | [Link](https://arxiv.org/abs/2102.07164) | [Link](https://github.com/virresh/viking) |
| 2021 | **GraphAttacker: A General Multi-Task Graph Attack Framework**  | Attack  |  Node Classification, Graph Classification, Link Prediction  | GNNs | Arxiv | [Link](https://arxiv.org/abs/2101.06855) |  |
| 2021 | **Membership Inference Attack on Graph Neural Networks**  | Attack  |  Membership Inference | GNNs | Arxiv | [Link](https://arxiv.org/abs/2101.06570) |  |


### Attack Papers 2020 [[Back to Top](#graph-adversarial-learning-literature)]
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Paper        |  Code      |
|-------|--------|--------|--------|-----------|------------|---------------|---------|
| 2020 | **Adversarial Label-Flipping Attack and Defense for Graph Neural Networks**  | Attack  |  Node Classification  |  GNNs | ICDM 2020 | [Link](http://shichuan.org/doc/97.pdf) | [Link](https://github.com/MengmeiZ/LafAK) |
| 2020 | **Exploratory Adversarial Attacks on Graph Neural Networks**  | Attack  |  Node Classification  | GCN | ICDM 2020 | [Link](https://ieeexplore.ieee.org/document/9338329) | [Link](https://github.com/EpoAtk/EpoAtk) |
| 2020 | **A Targeted Universal Attack on Graph Convolutional Network**  | Attack  |  Node Classification | GCN | Arxiv | [Link](https://arxiv.org/abs/2011.14365) | [Link](https://github.com/Nanyuu/TUA)  | 
| 2020 | **Attacking Graph-Based Classification without Changing Existing Connections**  | Attack  |  Node Classification  | Collective Classification Models | ACSAC 2020 | [Link](https://cse.sc.edu/~zeng1/papers/2020-acsac-graph.pdf) |  |
| 2020 | **Learning to Deceive Knowledge Graph Augmented Models via Targeted Perturbation**  | Attack  |  Commonsense Reasoning Recommender System  | Knowledge Graph | ICLR 2021 | [Link](https://arxiv.org/abs/2010.12872) | [Link](https://github.com/INK-USC/deceive-KG-models) |
| 2020 | **One Vertex Attack on Graph Neural Networks-based Spatiotemporal Forecasting**  | Attack  |  Spatiotemporal Forecasting  | GNNs | ICLR 2021 OpenReview | [Link](https://openreview.net/forum?id=W0MKrbVOxtd) |  |
| 2020 | **Single-Node Attack for Fooling Graph Neural Networks**  | Attack  |  Node Classification  | GNNs | ICLR 2021 OpenReview | [Link](https://openreview.net/forum?id=u4WfreuXxnk) |  |
| 2020 | **Black-Box Adversarial Attacks on Graph Neural Networks as An Influence Maximization Problem**  | Attack  |  Node Classification  | GNNs | ICLR 2021 OpenReview | [Link](https://openreview.net/forum?id=sbyjwhxxT8K) |  |
| 2020 | **Adversarial Attacks on Deep Graph Matching**  | Attack  |  Graph Matching  | Deep Graph Matching Models | NeurIPS 2020 | [Link](https://papers.nips.cc/paper/2020/file/ef126722e64e98d1c33933783e52eafc-Paper.pdf) |  |
| 2020 | **Towards More Practical Adversarial Attacks on Graph Neural Networks**  | Attack  |  Node Classification  | GNNs | NeurIPS 2020 | [Link](https://arxiv.org/abs/2006.05057) | [Link](https://github.com/Mark12Ding/GNN-Practical-Attack) |
| 2020 | **A Graph Matching Attack on Privacy-Preserving Record Linkage**  | Attack  |  Record Linkage  | Rrivacy-preserving Record Linkage Methods | CIKM 2020 | [Link](https://dl.acm.org/doi/abs/10.1145/3340531.3411931) |   |
| 2020 | **Adaptive Adversarial Attack on Graph Embedding via GAN**  | Attack  |  Node Classification  | GCN, DeepWalk, LINE | SocialSec | [Link](https://link.springer.com/chapter/10.1007/978-981-15-9031-3_7) |  |
| 2020 | **Scalable Adversarial Attack on Graph Neural Networks with Alternating Direction Method of Multipliers**  | Attack  |  Node Classification | GNNs | Arxiv | [Link](https://arxiv.org/abs/2009.10233) |  | 
| 2020 | **Semantic-preserving Reinforcement Learning Attack Against Graph Neural Networks for Malware Detection**  | Attack  |  Malware Detection | GCN | Arxiv | [Link](https://arxiv.org/abs/2009.05602) |  | 
| 2020 | **Adversarial Attack on Large Scale Graph**  | Attack  |  Node Classification | GNN | Arxiv | [Link](https://arxiv.org/abs/2009.03488) |  | 
| 2020 | **Efficient Evasion Attacks to Graph Neural Networks via Influence Function**  | Attack  |  Node Classification | GNN | Arxiv | [Link](https://arxiv.org/abs/2009.00203) |  | 
| 2020 | **Reinforcement Learning-based Black-Box Evasion Attacks to Link Prediction in Dynamic Graphs**  | Attack  |  Link Prediction | DyGCN | Arxiv | [Link](https://arxiv.org/abs/2009.00163) |  | 
| 2020 | **Adversarial attack on BC classification for scale-free networks**  | Attack  |  Broido and Clauset classification | scale-free network | AIP Chaos | [Link](https://aip.scitation.org/doi/full/10.1063/5.0003707) |  | 
| 2020 | **Adversarial Attacks on Link Prediction Algorithms Based on Graph Neural Networks**  | Attack  |  Link Prediction | GNN | Asia CCS 2020 | [Link](https://iqua.ece.toronto.edu/papers/wlin-asiaccs20.pdf) |  |  
| 2020 | **Practical Adversarial Attacks on Graph Neural Networks**  | Attack  |  Node Classification  | GNN | ICML 2020 Workshop | [Link](https://grlplus.github.io/papers/8.pdf) |  |
| 2020 | **Link Prediction Adversarial Attack Via Iterative Gradient Attack**  | Attack  |  Link Prediction | GAE | IEEE TCSS | [Link](https://ieeexplore.ieee.org/abstract/document/9141291?casa_token=JY86mKguq68AAAAA:GNbeDZJNuMzzcHFPGOTACf9ihXxgQyAOSjVUnbWhiON6vVG7ap7k8Ey4DCNyJTO0qlSxMyJWSY4B) |  |
| 2020 | **An Efficient Adversarial Attack on Graph Structured Data**  | Attack  |  Node Classification  | GCN | IJCAI 2020 Workshop | [Link](https://www.aisafetyw.org/programme) |  |
| 2020 | **Graph Backdoor**  | Attack  |  Node Classification Graph Classification  | GNNs | USENIX Security 2021 | [Link](https://arxiv.org/abs/2006.11890) |  |
| 2020 | **Backdoor Attacks to Graph Neural Networks**  | Attack  |  Graph Classification  | GNNs | Arxiv | [Link](https://arxiv.org/abs/2006.11165) |  |
| 2020 | **Robust Spammer Detection by Nash Reinforcement Learning**  | Attack  |  Fraud Detection  | Graph-based Fraud Detector | KDD 2020 | [Link](https://arxiv.org/abs/2006.06069) | [Link](https://github.com/YingtongDou/Nash-Detect) |
| 2020 | **Adversarial Attacks on Graph Neural Networks: Perturbations and their Patterns**  | Attack  |   Node Classification   | GNN | TKDD | [Link](https://dl.acm.org/doi/10.1145/3394520) | |
| 2020 | **Adversarial Attack on Hierarchical Graph Pooling Neural Networks**  | Attack  |  Graph Classification  | GNN | Arxiv | [Link](https://arxiv.org/abs/2005.11560) | |
| 2020 | **Stealing Links from Graph Neural Networks**  | Attack  |  Inferring Link  | GNNs | USENIX Security 2021 | [Link](https://www.usenix.org/system/files/sec21summer_he.pdf) | |
| 2020 | **Scalable Attack on Graph Data by Injecting Vicious Nodes**  | Attack  |  Node Classification  | GCN | ECML-PKDD 2020 | [Link](https://arxiv.org/abs/2004.13825) | |
| 2020 | **Network disruption: maximizing disagreement and polarization in social networks**  | Attack  |  Manipulating Opinion  | Graph Model, Social Network | Arxiv | [Link](https://arxiv.org/abs/2003.08377) | |
| 2020 | **Adversarial Perturbations of Opinion Dynamics in Networks**  | Attack  |  Manipulating Opinion  | Graph Model | Arxiv | [Link](https://arxiv.org/abs/2003.07010) | |
| 2020 | **Non-target-specific Node Injection Attacks on Graph Neural Networks: A Hierarchical Reinforcement Learning Approach**  | Attack  |  Node Classification  | GCN | WWW 2020 | [Link](https://faculty.ist.psu.edu/vhonavar/Papers/www20.pdf) | |
| 2020 | **MGA: Momentum Gradient Attack on Network**  | Attack  |  Node Classification, Community Detection | GCN, DeepWalk, node2vec | Arxiv | [Link](https://arxiv.org/abs/2002.11320) |     |
| 2020 | **Indirect Adversarial Attacks via Poisoning Neighbors for Graph Convolutional Networks**  | Attack  |  Node Classification | GCN | BigData 2019 | [Link](https://arxiv.org/abs/2002.08012) |    |
| 2020 | **Graph Universal Adversarial Attacks: A Few Bad Actors Ruin Graph Learning Models**  | Attack  |  Node Classification | GCN | Arxiv | [Link](https://arxiv.org/abs/2002.04784) |   [Link](https://github.com/chisam0217/Graph-Universal-Attack)  |
| 2020 | **Adversarial Attacks to Scale-Free Networks: Testing the Robustness of Physical Criteria**  | Attack  | Network Structure  | Physical Criteria | Arxiv | [Link](https://arxiv.org/abs/2002.01249) |      |
| 2020 | **Adversarial Attack on Community Detection by Hiding Individuals**  | Attack  |  Community Detection  | GCN | WWW 2020 | [Link](https://arxiv.org/abs/2001.07933) |[Link](https://github.com/halimiqi/CD-ATTACK) |

### Attack Papers 2019 [[Back to Top](#graph-adversarial-learning-literature)]
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Paper        |  Code      |
|-------|--------|--------|--------|-----------|------------|---------------|---------|
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

### Attack Papers 2018 [[Back to Top](#graph-adversarial-learning-literature)]
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Paper        |  Code      |
|-------|--------|--------|--------|-----------|------------|---------------|---------|
| 2018 | **Poisoning Attacks to Graph-Based Recommender Systems**  | Attack  | Recommender System   | Graph-based Recommendation Algs | ACSAC 2018| [Link](https://arxiv.org/abs/1809.04127) |      |
| 2018 | **GA Based Q-Attack on Community Detection**  | Attack  | Community Detection   | Modularity, Community Detection Alg | IEEE TCSS| [Link](https://ieeexplore.ieee.org/abstract/document/8714065) |      |
| 2018 | **Data Poisoning Attack against Unsupervised Node Embedding Methods**  | Attack  | Link Prediction   | LINE, DeepWalk | Arxiv| [Link](https://arxiv.org/abs/1810.12881) |   |
| 2018 | **Attack Graph Convolutional Networks by Adding Fake Nodes**  | Attack  | Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1810.10751) |     |
| 2018 | **Link Prediction Adversarial Attack**  | Attack  | Link Prediction   | GAE, GCN | Arxiv | [Link](https://arxiv.org/abs/1810.01110) |      |
| 2018 | **Attack Tolerance of Link Prediction Algorithms: How to Hide Your Relations in a Social Network**  | Attack  | Link Prediction   | Traditional Link Prediction Algs | Scientific Reports | [Link](https://arxiv.org/abs/1809.00152) |       |
| 2018 | **Attacking Similarity-Based Link Prediction in Social Networks**  | Attack  | Link Prediction   | local&global similarity metrics | AAMAS 2019 | [Link](https://dl.acm.org/citation.cfm?id=3306127.3331707) |    |
| 2018 | **Fast Gradient Attack on Network Embedding**  | Attack  | Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1809.02797) |     |
| 2018 | **Adversarial Attack on Graph Structured Data**  | Attack  | Node Classification, Graph Classification   | GNN, GCN | ICML 2018 | [Link](https://arxiv.org/abs/1806.02371) |  [Link](https://github.com/Hanjun-Dai/graph_adversarial_attack)   |
| 2018 | **Adversarial Attacks on Neural Networks for Graph Data**  | Attack  | Node Classification   | GCN | KDD 2018 | [Link](https://arxiv.org/abs/1805.07984) | [Link](https://github.com/danielzuegner/nettack) |
| 2018 | **Hiding individuals and communities in a social network**  | Attack  | Community Detection   | Community Detection Algs | Nature Human Behavior | [Link](https://arxiv.org/abs/1608.00375) |  [Link](https://github.com/DSE-MSU/DeepRobust)   |

### Attack Papers 2017 [[Back to Top](#graph-adversarial-learning-literature)]
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Paper        |  Code      |
|-------|--------|--------|--------|-----------|------------|---------------|---------|
| 2017 | **Practical Attacks Against Graph-based Clustering**  | Attack  | Graph Clustering   | SVD, node2vec, Community Detection Alg | CCS 2017| [Link](https://arxiv.org/abs/1708.09056) |      |
| 2017 | **Adversarial Sets for Regularising Neural Link Predictors**  | Attack  | Link Prediction   | Knowledge Graph Embeddings | UAI 2017 | [Link](https://arxiv.org/abs/1707.07596) |  [Link](https://github.com/uclmr/inferbeddings)   |

## Defense

### Defense Papers 2021 [[Back to Top](#graph-adversarial-learning-literature)]
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Paper        |  Code |
|-------|--------|--------|--------|-----------|------------|---------------|-------|
| 2021 | **Expressive 1-Lipschitz Neural Networks for Robust Multiple Graph Learning against Adversarial Attacks**  | Defense  |   |   | ICML 2021 | Link |  |
| 2021 | **Integrated Defense for Resilient Graph Matching**  | Defense  | Graph Matching  |   | ICML 2021 | Link |  |
| 2021 | **NetFense: Adversarial Defenses against Privacy Attacks on Neural Networks for Graph Data**  | Defense  |  Privacy Protection  |  GNNs | TKDE | [Link](https://ieeexplore.ieee.org/abstract/document/9448513) |  |
| 2021 | **Stability of graph convolutional neural networks to stochastic perturbations**  | Defense  |  Robustness Certification  |  GNNs | Signal Processing | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0165168421002541) |  |
| 2021 | **DeepInsight: Interpretability Assisting Detection of Adversarial Samples on Graphs**  | Defense  | Node Classification  |  GNNs | Arxiv | [Link](https://arxiv.org/abs/2106.09501) |  |
| 2021 | **Improving Robustness of Graph Neural Networks with Heterophily-Inspired Designs**  | Defense  | Node Classification  |  GNNs | Arxiv | [Link](https://arxiv.org/abs/2106.07767) |  |
| 2021 | **Understanding Structural Vulnerability in Graph Convolutional Networks**  | Defense  | Node Classification  |  GNNs | IJCAI 2021 | [Link](cs.emory.edu/~jyang71/files/rpgcn.pdf) | [Link](https://github.com/EdisonLeeeee/MedianGCN) |
| 2021 | **Certified Robustness of Graph Neural Networks against Adversarial Structural Perturbation**  | Defense  | Robustness Certification  |  GNNs | KDD 2021 | [Link](https://arxiv.org/abs/2008.10715) |  |
| 2021 | **Unveiling Anomalous Nodes Via Random Sampling and Consensus on Graphs**  | Defense  | Anomaly Detection   | Anomaly Detection Algs | ICASSP 2021 | [Link](https://ieeexplore.ieee.org/abstract/document/9414953) |  |
| 2021 | **Graph Sanitation with Application to Node Classification**  | Defense  | Node Classification  |  GNNs | Arxiv | [Link](https://arxiv.org/pdf/2105.09384.pdf) |  |
| 2021 | **Robust Network Alignment via Attack Signal Scaling and Adversarial Perturbation Elimination**  | Defense  | Network Alignment  |  Network Alignment Algorithms | WWW 2021 | [Link](http://eng.auburn.edu/users/yangzhou/papers/RNA.pdf) |  |
| 2021 | **Information Obfuscation of Graph Neural Networks**  | Defense  | Recommender System, Knowledge Graph, Quantum Chemistry  | GNNs | ICML 2021 | [Link](https://arxiv.org/pdf/2009.13504.pdf) | [Link](https://github.com/liaopeiyuan/GAL) |
| 2021 | **Graph Embedding for Recommendation against Attribute Inference Attacks**  | Defense  | Recommender System  | GCN | WWW 2021 | [Link](https://arxiv.org/pdf/2101.12549.pdf) |  |
| 2021 | **Spatio-Temporal Sparsification for General Robust Graph Convolution Networks**  | Defense  | Node Classification  | GCN | Arxiv | [Link](https://arxiv.org/abs/2103.12256) |  |
| 2021 | **Detection and Defense of Topological Adversarial Attacks on Graphs**  | Defense  | Node Classification  | GCN | AISTATS 2021 | [Link](http://proceedings.mlr.press/v130/zhang21i.html) |  |
| 2021 | **Robust graph convolutional networks with directional graph adversarial training**  | Defense  | Node Classification  | GCN | Applied Intelligence | [Link](https://link.springer.com/article/10.1007/s10489-021-02272-y) |  |
| 2021 | **Interpretable Stability Bounds for Spectral Graph Filters**  | Defense  | Robustness Certification  | Spectral Graph Filter | Arxiv | [Link](https://arxiv.org/abs/2102.09587) |  |
| 2021 | **Personalized privacy protection in social networks through adversarial modeling**  | Defense  | Privacy Protection  | GCN | AAAI 2021 | [Link](https://www.cs.uic.edu/~elena/pubs/biradar-ppai21.pdf) |  |
| 2021 | **Node Similarity Preserving Graph Convolutional Networks**  | Defense  | Node Classification | GNNs | WSDM 2021 | [Link](https://arxiv.org/abs/2011.09643) | [Link](https://github.com/ChandlerBang/SimP-GCN) |

### Defense Papers 2020 [[Back to Top](#graph-adversarial-learning-literature)]
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Paper        |  Code |
|-------|--------|--------|--------|-----------|------------|---------------|-------|
| 2020 | **Smoothing Adversarial Training for GNN**  | Defense  |  Node Classification, Community Detection  |  GCN | IEEE TCSS | [Link](https://ieeexplore.ieee.org/abstract/document/9305289?casa_token=fTXIL3hT1yIAAAAA:I4fn-GlF0PIwzPRC87SayRi5_pi2ZDDuSancEsY96A4O4bUBEsp0hSYMNJVGVzMgBWxycYN9qu6D) |  |
| 2020 | **Unsupervised Adversarially-Robust Representation Learning on Graphs**  | Defense  |  Node Classification  |  GNNs | Arxiv | [Link](https://arxiv.org/abs/2012.02486) |  |
| 2020 | **AANE: Anomaly Aware Network Embedding For Anomalous Link Detection**  | Defense  |  Node Classification  |  GNNs | ICDM 2020 | [Link](https://ieeexplore.ieee.org/document/9338406) |  |
| 2020 | **Provably Robust Node Classification via Low-Pass Message Passing**  | Defense  |  Anomaly Detection  |  GNNs | ICDM 2020 | [Link](https://shenghua-liu.github.io/papers/icdm2020-provablerobust.pdf) |  |
| 2020 | **Learning to Drop: Robust Graph Neural Network via Topological Denoising**  | Defense  | Node Classification | GNNs | WSDM 2021 | [Link](https://arxiv.org/abs/2011.07057) | [Link](https://github.com/flyingdoog/PTDNet) |
| 2020 | **Robust Android Malware Detection Based on Attributed Heterogenous Graph Embedding**  | Defense  | Malware Detection | Heterogeneous Information Network Embedding | FCS 2020 | [Link](https://link.springer.com/chapter/10.1007/978-981-15-9739-8_33) |  |
| 2020 | **Adversarial Detection on Graph Structured Data**  | Defense  | Graph Classification  |  GNNs | PPMLP 2020 | [Link](https://dl.acm.org/doi/abs/10.1145/3411501.3419424) |  |
| 2020 | **On the Stability of Graph Convolutional Neural Networks under Edge Rewiring**  | Defense  |  Robustness Certification  |  GNNs | Arxiv | [Link](https://arxiv.org/abs/2010.13747) |  |
| 2020 | **Collective Robustness Certificates**  | Defense  |  Robustness Certification  | GNNs | ICLR 2021 | [Link](https://openreview.net/forum?id=ULQdiUTHe3y) |  |
| 2020 | **Towards Robust Graph Neural Networks against Label Noise**  | Defense  |  Node Classification  | GNNs | ICLR 2021 OpenReview | [Link](https://openreview.net/forum?id=H38f_9b90BO) |  |
| 2020 | **Certifying Robustness of Graph Laplacian Based Semi-Supervised Learning**  | Defense  |  Robustness Certification  | GNNs | ICLR 2021 OpenReview | [Link](https://openreview.net/forum?id=cQyybLUoXxc) |  |
| 2020 | **Graph Adversarial Networks: Protecting Information against Adversarial Attacks**  | Defense  |  Node Attribute Inference  | GNNs | ICLR 2021 OpenReview | [Link](https://openreview.net/forum?id=Q8ZdJahesWe) |  |
| 2020 | **Ricci-GNN: Defending Against Structural Attacks Through a Geometric Approach**  | Defense  |  Node Classification  | GNNs | ICLR 2021 OpenReview | [Link](https://openreview.net/forum?id=_qoQkWNEhS) |  |
| 2020 | **Graph Contrastive Learning with Augmentations**  | Defense  |  Node Classification  |  GNNs | NeurIPS 2020 | [Link](https://arxiv.org/abs/2010.13902) | [Link](https://github.com/Shen-Lab/GraphCL) |
| 2020 | **Graph Information Bottleneck**  | Defense  |  Node Classification  |  GNNs | NeurIPS 2020 | [Link](https://arxiv.org/abs/2010.12811) | [Link](https://github.com/snap-stanford/GIB) |
| 2020 | **Certified Robustness of Graph Convolution Networks for Graph Classification under Topological Attacks**  | Defense  |  Graph Classification  |  GCN | NeurIPS 2020 | [Link](https://www.cs.uic.edu/~zhangx/papers/Jinetal20.pdf) | [Link](https://github.com/RobustGraph/RoboGraph) |
| 2020 | **Reliable Graph Neural Networks via Robust Aggregation**  | Defense  |  Node Classification  | GNNs | NeurIPS 2020 | [Link](https://arxiv.org/abs/2010.15651) | [Link](https://github.com/sigeisler/reliable_gnn_via_robust_aggregation) |
| 2020 | **Graph Random Neural Networks for Semi-Supervised Learning on Graphs**  | Defense  |  Node Classification  |  GCN | NeurIPS 2020 | [Link](https://arxiv.org/abs/2005.11079) | [Link](https://github.com/Grand20/grand)  |
| 2020 | **Variational Inference for Graph Convolutional Networks in the Absence of Graph Data and Adversarial Settings**  | Defense  |  Node Classification  |  GCN | NeurIPS 2020 | [Link](https://arxiv.org/abs/1906.01852) | [Link](https://github.com/ebonilla/VGCN) |
| 2020 | **GNNGuard: Defending Graph Neural Networks against Adversarial Attacks**  | Defense  |   Node Classification | GNNs | NeurIPS 2020 | [Link](https://arxiv.org/abs/2006.08149) | [Link](https://github.com/mims-harvard/GNNGuard) |
| 2020 | **A Feature-Importance-Aware and Robust Aggregator for GCN**  | Defense  |  Node Classification Graph Classification  |  GNNs | CIKM 2020 | [Link](https://dl.acm.org/doi/abs/10.1145/3340531.3411983) | [Link](https://github.com/LiZhang-github/LA-GCN) |
| 2020 | **Uncertainty-Matching Graph Neural Networks to Defend Against Poisoning Attacks**  | Defense  |  Node Classification  |  GNNs | AAAI 2021 | [Link](https://arxiv.org/abs/2009.14455) |  |
| 2020 | **Cross Entropy Attack on Deep Graph Infomax**  | Defense  |  Node Classification  |  DGI | IEEE ISCAS | [Link](https://ieeexplore.ieee.org/document/9180817) |  |
| 2020 | **RoGAT: a robust GNN combined revised GAT with adjusted graphs**  | Defense  |  Node Classification  |  GNNs | Arxiv | [Link](https://arxiv.org/abs/2009.13038) |  |
| 2020 | **A Novel Defending Scheme for Graph-Based Classification Against Graph Structure Manipulating Attack**  | Defense  |  Node Classification  |  MRF | SocialSec | [Link](https://link.springer.com/chapter/10.1007/978-981-15-9031-3_26) |  |
| 2020 | **Uncertainty-aware Attention Graph Neural Network for Defending Adversarial Attacks**  | Defense  |  Node Classification  |  GNNs | AAAI 2021 | [Link](https://arxiv.org/abs/2009.10235) |  |
| 2020 | **Certified Robustness of Graph Classification against Topology Attack with Randomized Smoothing**  | Defense  |  Graph Classification  |  GCB | IEEE GLOBECOM 2020 | [Link](https://arxiv.org/abs/2009.05872) |  |
| 2020 | **Adversarial Immunization for Improving Certifiable Robustness on Graphs**  | Defense  |  Node Classification  |  GNNs | WSDM 2021 | [Link](https://arxiv.org/abs/2007.09647) |  |
| 2020 | **Robust Collective Classification against Structural Attacks**  | Defense  |  Node Classification  |  Associative Markov Networks | UAI 2020 | [Link](http://www.auai.org/uai2020/proceedings/119_main_paper.pdf) |  |
| 2020 | **Enhancing Robustness of Graph Convolutional Networks via Dropping Graph Connections**  | Defense  |  Node Classification  | GCN | Preprint | [Link](https://faculty.ist.psu.edu/wu/papers/DropCONN.pdf) |  |
| 2020 | **Robust Training of Graph Convolutional Networks via Latent Perturbation**  | Defense  |  Node Classification  | GCN | ECML-PKDD 2020 | [Link](https://www.cs.uic.edu/~zhangx/papers/JinZha20.pdf) |  |
| 2020 | **Backdoor Attacks to Graph Neural Networks**  | Defense  |  Graph Classification  | GNNs | Arxiv | [Link](https://arxiv.org/abs/2006.11165) |  |
| 2020 | **DefenseVGAE: Defending against Adversarial Attacks on Graph Data via a Variational Graph Autoencoder**  | Defense  |  Node Classification | GNNs | Arxiv | [Link](https://arxiv.org/abs/2006.08900) | [Link](https://github.com/zhangao520/defense-vgae) |
| 2020 | **Robust Spammer Detection by Nash Reinforcement Learning**  | Defense  |  Fraud Detection  | Graph-based Fraud Detector | KDD 2020 | [Link](https://arxiv.org/abs/2006.06069) | [Link](https://github.com/YingtongDou/Nash-Detect) |
| 2020 | **Certifiable Robustness of Graph Convolutional Networks under Structure Perturbations**  | Defense  |  Robustness Certification  | GCN | KDD 2020 | [Link](https://dl.acm.org/doi/abs/10.1145/3394486.3403217) | [Link](https://github.com/danielzuegner/robust-gcn-structure) |
| 2020 | **Efficient Robustness Certificates for Discrete Data: Sparsity-Aware Randomized Smoothing for Graphs, Images and More**  | Defense  |  Robustness Certification  | GNN | ICML 2020 | [Link](https://proceedings.icml.cc/static/paper_files/icml/2020/6890-Paper.pdf) | [Link](https://github.com/abojchevski/sparse_smoothing) |
| 2020 | **Robust Graph Representation Learning via Neural Sparsification**  | Defense  |  Node Classification  | GNN | ICML 2020 | [Link](https://proceedings.icml.cc/static/paper_files/icml/2020/2611-Paper.pdf) |   |
| 2020 | **EDoG: Adversarial Edge Detection For Graph Neural Networks**  | Defense  | Edge Detection  | GCN |Preprint | [Link](https://www.osti.gov/servlets/purl/1631086) | |
| 2020 | **Graph Structure Learning for Robust Graph Neural Networks**  | Defense  |  Node Classification  | GCN | KDD 2020 | [Link](https://arxiv.org/abs/2005.10203) |[Link](https://github.com/DSE-MSU/DeepRobust) |
| 2020 | **GCN-Based User Representation Learning for Unifying Robust Recommendation and Fraudster Detection**  | Defense  |  Recommender System  | GCN | SIGIR 2020 | [Link](https://arxiv.org/abs/2005.10150) ||
| 2020 | **Anonymized GCN: A Novel Robust Graph Embedding Method via Hiding Node Position in Noise**  | Defense  |  Node Classification | GCN | Arxiv | [Link](https://arxiv.org/abs/2005.03482) |    |
| 2020 | **A Robust Hierarchical Graph Convolutional Network Model for Collaborative Filtering**  | Defense  |  Recommender System | GCN | Arxiv | [Link](https://arxiv.org/abs/2004.14734) |    |
| 2020 | **On The Stability of Polynomial Spectral Graph Filters**  | Defense  |  Graph Property | Spectral Graph Filter | ICASSP 2020 | [Link](https://ieeexplore.ieee.org/abstract/document/9054072) |  [Link](https://github.com/henrykenlay/spgf)  |
| 2020 | **On the Robustness of Cascade Diffusion under Node Attacks**  | Defense  |  Influence Maximization | IC Model | WWW 2020 Workshop | [Link](https://www.cs.au.dk/~karras/robustIC.pdf) |  [Link](https://github.com/allogn/robustness)   |
| 2020 | **Friend or Faux: Graph-Based Early Detection of Fake Accounts on Social Networks**  | Defense  | Fraud Detection | Graph-based Fraud Detectors | WWW 2020 | [Link](https://arxiv.org/abs/2004.04834) |    |
| 2020 | **Tensor Graph Convolutional Networks for Multi-relational and Robust Learning**  | Defense  |  Node Classification | GCN | Arxiv | [Link](https://arxiv.org/abs/2003.07729) |     |
| 2020 | **Adversary for Social Good: Protecting Familial Privacy through Joint Adversarial Attacks**  | Defense  |  Node Classification  | Privacy Protection | AAAI 2020 | [Link](https://ojs.aaai.org//index.php/AAAI/article/view/6791) | |
| 2020 | **Improving the Robustness of Wasserstein Embedding by Adversarial PAC-Bayesian Learning**  | Defense  |  Robustness Certification  |  Wasserstein Embedding | AAAI 2020 | [Link](http://staff.ustc.edu.cn/~hexn/papers/aaai20-adversarial-embedding.pdf) | |
| 2020 | **Adversarial Perturbations of Opinion Dynamics in Networks**  | Defense  |  Manipulating Opinion  | Graph Model | Arxiv | [Link](https://arxiv.org/abs/2003.07010) | |
| 2020 | **Topological Effects on Attacks Against Vertex Classification**  | Defense  |  Node Classification | GCN | Arxiv | [Link](https://arxiv.org/abs/2003.05822) |     |
| 2020 | **Towards an Efficient and General Framework of Robust Training for Graph Neural Networks**  | Defense  |  Node Classification | GCN | ICASSP 2020 | [Link](https://arxiv.org/abs/2002.10947) |    |
| 2020 | **Certified Robustness of Community Detection against Adversarial Structural Perturbation via Randomized Smoothing**  | Defense  |  Community Detection  | Community Detection Algs | WWW 2020 | [Link](https://arxiv.org/abs/2002.03421) |      |
| 2020 | **Data Poisoning Attacks on Graph Convolutional Matrix Completion**  | Defense  |  Recommender System  | GCMC | ICA3PP 2019 | [Link](https://link.springer.com/chapter/10.1007/978-3-030-38961-1_38) |      |

### Defense Papers 2019 [[Back to Top](#graph-adversarial-learning-literature)]
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Paper        |  Code |
|-------|--------|--------|--------|-----------|------------|---------------|-------|
| 2019 | **How Robust Are Graph Neural Networks to Structural Noise?**  | Defense  |  Node Structural Identity Prediction | GIN | Arxiv | [Link](https://arxiv.org/abs/1912.10206) |      |
| 2019 | **GraphDefense: Towards Robust Graph Convolutional Networks**  | Defense  |  Node Classification | GCN | Arxiv | [Link](https://arxiv.org/abs/1911.04429) |     |
| 2019 | **All You Need is Low (Rank): Defending Against Adversarial Attacks on Graphs**  | Defense  |  Node Classification | GCN, Tensor Embedding | WSDM 2020 | [Link](https://dl.acm.org/doi/abs/10.1145/3336191.3371789) | [Link](https://github.com/DSE-MSU/DeepRobust)    |
| 2019 | **αCyber: Enhancing Robustness of Android Malware Detection System against Adversarial Attacks on Heterogeneous Graph based Model**  | Defense  | Malware Detection  | HIN | CIKM 2019 | [Link](https://dl.acm.org/citation.cfm?id=3357875) |      |
| 2019 | **Edge Dithering for Robust Adaptive Graph Convolutional Networks**  | Defense  |  Node Classification   | GCN | Arxiv | [Link](https://arxiv.org/abs/1910.09590) |        |
| 2019 | **GraphSAC: Detecting anomalies in large-scale graphs**  | Defense  |  Anomaly Detection   | Anomaly Detection Algs | Arxiv | [Link](https://arxiv.org/abs/1910.09589) |    |
| 2019 | **Certifiable Robustness to Graph Perturbations**  | Defense  | Robustness Certification  | GNN | NeurIPS 2019 | [Link](https://papers.nips.cc/paper/9041-certifiable-robustness-to-graph-perturbations.pdf) |   [Link](https://github.com/abojchevski/graph_cert)  |
| 2019 | **Power up! Robust Graph Convolutional Network based on Graph Powering**  | Defense  |  Node Classification   | GCN | Openreview | [Link](https://openreview.net/pdf?id=BkxDxJHFDr) |  [Link](https://www.dropbox.com/sh/p36pzx1ock2iamo/AABEr7FtM5nqwC4i9nICLIsta?dl=0)   |
| 2019 | **Adversarial Robustness of Similarity-Based Link Prediction**  | Defense  |  Link Prediction   | Local Similarity Metrics | ICDM 2019 | [Link](https://arxiv.org/abs/1909.01432) |       |
| 2019 | **Adversarial Training Methods for Network Embedding**  | Defense  |  Node Classification   | DeepWalk | WWW 2019 | [Link](https://arxiv.org/abs/1908.11514) |   [Link](https://github.com/wonniu/AdvT4NE_WWW2019)    |
| 2019 | **Transferring Robustness for Graph Neural Network Against Poisoning Attacks**  | Defense  |  Node Classification   | GNN | WSDM 2020 | [Link](https://arxiv.org/abs/1908.07558) |   [Link](https://github.com/tangxianfeng/PA-GNN)    |
| 2019 | **Improving Robustness to Attacks Against Vertex Classification**  | Defense  |  Node Classification   | GCN | KDD Workshop 2019 | [Link](http://eliassi.org/papers/benmiller-mlg2019.pdf) |       |
| 2019 | **Target Defense Against Link-Prediction-Based Attacks via Evolutionary Perturbations**  | Defense  |  Link Prediction   | Link Prediction Algs | TKDE | [Link](https://arxiv.org/abs/1809.05912) |     |
| 2019 | **Latent Adversarial Training of Graph Convolution Networks**  | Defense  |  Node Classification   | GCN | LRGSD@ICML | [Link](https://graphreason.github.io/papers/35.pdf) |     |
| 2019 | **Certifiable Robustness and Robust Training for Graph Convolutional Networks**  | Defense  |  Robustness Certification   | GCN | KDD 2019 | [Link](https://arxiv.org/abs/1906.12269) |  [Link](https://github.com/danielzuegner/robust-gcn)   |
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

### Defense Papers 2018 [[Back to Top](#graph-adversarial-learning-literature)]
| Year        | Title           | Type       |  Target Task | Target Model     | Venue    | Paper        |  Code |
|-------|--------|--------|--------|-----------|------------|---------------|-------|
| 2018 | **Characterizing Malicious Edges targeting on Graph Neural Networks**  | Defense  | Detected Added Edges   | GNN, GCN |  OpenReview | [Link](https://openreview.net/forum?id=HJxdAoCcYX) |       |
| 2018 | **PeerNets: Exploiting Peer Wisdom Against Adversarial Attacks**  | Defense  | Image Classification   | LeNet, ResNet |  ICLR 2019 | [Link](https://arxiv.org/abs/1806.00088) |       |
