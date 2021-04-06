---
layout: archive
title: ""
permalink: /blog/quaternion-graph-neural-networks 
author_profile: true
---


# Quaternion Graph Neural Networks 
<sub>Dai Quoc Nguyen, 25 September, 2020 </sub>

<p align="center">
	<img src="https://raw.githubusercontent.com/daiquocnguyen/daiquocnguyen.github.io/master/_pages/qgnn.png" width="550">
</p>

1. [Introduction](#gnns)
2. [Quaternion Background](#background)
3. [Quaternion Graph Neural Networks](#qgnn)
4. [Conclusion](#conclusion)

## Introduction<a name="gnns"></a>

Recently, graph neural network (GNN)-based approaches become a principal research direction to learn low-dimensional continuous embeddings of nodes and graphs to predict node and graph labels, respectively.
In general, GNNs use an <i>Aggregation</i> function [1, 2, 3] over neighbors of each node to update its vector representation iteratively. 
After that, GNNs utilize a <i>ReadOut</i> pooling function to obtain graph embeddings [4, 5, 6, 8].

Mathematically, given a graph G = (V, E, {<b>h</b><sub>v</sub>}<sub>∀v∈V</sub>), where V is a set of nodes, E is a set of edges, and <b>h</b><sub>v</sub> is the Euclidean feature vector of node v ∈ V, we have:

<p align="center"> <b>h</b><sub>v</sub><sup>(l+1)</sup> = <i>Aggregation</i>({<b>h</b><sub>u</sub><sup>(l)</sup>}<sub>u∈N<sub>v</sub>∪{v}</sub>) </p>

where <b>h</b><sub>v</sub><sup>(l)</sup> is the vector representation of node v at the <i>l</i>-th iteration/layer, N<sub>v</sub> is the set of neighbors of node v, and <b>h</b><sub>v</sub><sup>(0)</sup> = <b>h</b><sub>v</sub>.

There have been many designs for the <i>Aggregation</i> functions proposed in recent literature [1, 7]. The widely-used one is introduced in Graph Convolutional Network (GCN) [1] as:

<p align="center">  <b>h</b><sub>v</sub><sup>(l+1)</sup> = g(∑<sub>u∈N<sub>v</sub>∪{v}</sub> <i>a</i><sub>v,u</sub><b>W</b><sup>(l)</sup> <b>h</b><sub>u</sub><sup>(l)</sup>), ∀ v ∈ V </p>

where <i>a</i><sub>v,u</sub> is an edge constant between nodes v and u in the re-normalized adjacency matrix, <b>W</b><sup>(l)</sup> is a weight matrix, and g is a nonlinear activation function.

We follow [7] to employ a concatenation over the vector representations of node v at the different layers to construct the node embedding <b>e</b><sub>v</sub>.
The graph-level <i>ReadOut</i> function can be a simple sum pooling or a complex pooling such as sort pooling [5], hierarchical pooling [8], and differentiable pooling [6]. 
As the sum pooling produces competitive accuracies for graph classification task [7], we utilize the sum pooling to obtain the embedding <b>e</b><sub>G</sub> of the entire graph G as: <b>e</b><sub>G</sub> = ∑<sub>∀v∈V</sub> <b>e</b><sub>v</sub>.

While it has been considered under other contexts, in this blog, we address the following question: <i>Can we move beyond the Euclidean space to learn better graph representations?</i> To this end, we propose to learn quaternion embeddings for nodes and graphs and introduce a novel form of quaternion graph neural networks (QGNN) to generalize GCNs within the Quaternion space.

## Quaternion with Hamilton product<a name="background"></a>

Recently the use of hyper-complex vector space has considered on the Quaternion space [9] consisting of one real and three separate imaginary axes.
The Quaternion space has been applied to image classification [10, 11], speech recognition [12, 13], knowledge graph [14], and natural language processing [15].
We provide key notations and operations related to quaternion space required for later development.

A quaternion <i>q</i> ∈ H is a hyper-complex number con-sisting of a real and three separate imaginary components [9] defined as: 

<p align="center"> <i>q</i> = <i>q</i><sub>r</sub> + <i>q</i><sub>i</sub><b>i</b> + <i>q</i><sub>j</sub><b>j</b> + <i>q</i><sub>k</sub><b>k</b> </p>

where <i>q</i><sub>r</sub>, <i>q</i><sub>i</sub>, <i>q</i><sub>j</sub>, <i>q</i><sub>k</sub> ∈ R, and <b>i</b>, <b>j</b>, <b>k</b> are imaginary units that <b>ijk</b> = <b>i</b><sup>2</sup> = <b>j</b><sup>2</sup> = <b>k</b><sup>2</sup> = −1, leads to non-commutative multiplication rules as <b>ij</b> = <b>k</b>, <b>ji</b> = −<b>k</b>, <b>jk</b> = <b>i</b>, <b>kj</b> = −<b>i</b>, <b>ki</b> = <b>j</b>, and <b>ik</b> = −<b>j</b>.  Correspondingly, a <i>n</i>-dimensional quaternion vector <b><i>q</i></b> ∈ H<sup>n</sup> is defined as:

<p align="center"> <b><i>q</i></b> = <b><i>q</i></b><sub>r</sub> + <b><i>q</i></b><sub>i</sub><b>i</b> + <b><i>q</i></b><sub>j</sub><b>j</b> + <b><i>q</i></b><sub>k</sub><b>k</b> </p>

where <b><i>q</i></b><sub>r</sub>, <b><i>q</i></b><sub>i</sub>, <b><i>q</i></b><sub>j</sub>, <b><i>q</i></b><sub>k</sub> ∈ R<sup>n</sup>. The operations for the Quaternion algebra are defined as follows:

* <b>Conjugate.</b> The conjugate <i>q</i><sup>*</sup> of a quaternion <i>q</i> is defined as: <i>q</i><sup>*</sup> = <i>q</i><sub>r</sub> - <i>q</i><sub>i</sub><b>i</b> - <i>q</i><sub>j</sub><b>j</b> - <i>q</i><sub>k</sub><b>k</b>
  
* <b>Addition.</b> The addition of two quaternions <i>q</i> and <i>p</i> is defined as: <i>q</i> + <i>p</i> = (<i>q</i><sub>r</sub> + <i>p</i><sub>r</sub>) + (<i>q</i><sub>i</sub> + <i>p</i><sub>i</sub>)<b>i</b> + (<i>q</i><sub>j</sub> + <i>p</i><sub>j</sub>)<b>j</b> + (<i>q</i><sub>k</sub> + <i>p</i><sub>k</sub>)<b>k</b>
  
* <b>Scalar multiplication.</b> The multiplication of a scalar λ and a quaternion <i>q</i> is defined as: λ<i>q</i> = λ<i>q</i><sub>r</sub> + λ<i>q</i><sub>i</sub><b>i</b> + λ<i>q</i><sub>j</sub><b>j</b> + λ<i>q</i><sub>k</sub><b>k</b>

* <b>Norm.</b> The norm ‖<i>q</i>‖ of a quaternion <i>q</i> is defined as: ‖<i>q</i>‖ = <i>q</i><sub>r</sub><sup>2</sup> + <i>q</i><sub>i</sub><sup>2</sup> + <i>q</i><sub>j</sub><sup>2</sup> + <i>q</i><sub>k</sub><sup>2</sup>

* <b>Hamilton  product.</b> The  Hamilton  product ⊗ (i.e., the quaternion multiplication) of two quaternions <i>q</i> and <i>p</i> is defined as:

<p align="center">
	<img src="https://raw.githubusercontent.com/daiquocnguyen/daiquocnguyen.github.io/master/_pages/hamilton_product1.png" width="345">
</p>

* Note that the Hamilton product is not commutative, i.e., <i>q</i> ⊗ <i>p</i> ≠ <i>p</i> ⊗ <i>q</i>.

* <b>Concatenation.</b> We use a concatenation of two quaternion vectors <b><i>q</i></b> and <b><i>p</i></b> as: [<b><i>q</i></b>; <b><i>p</i></b>] = [<b><i>q</i></b><sub>r</sub>; <b><i>p</i></b><sub>r</sub>] + [<b><i>q</i></b><sub>i</sub>; <b><i>p</i></b><sub>i</sub>]<b>i</b> + [<b><i>q</i></b><sub>j</sub>; <b><i>p</i></b><sub>j</sub>]<b>j</b> + [<b><i>q</i></b><sub>k</sub>; <b><i>p</i></b><sub>k</sub>]<b>k</b>


## Quaternion Graph Neural Networks<a name="qgnn"></a>

In QGNN, the <i>Aggregation</i> function at the <i>l</i>-th layer is defined as:

<p align="center">  <b>h</b><sub>v</sub><sup>(l+1),Q</sup> = g(∑<sub>u∈N<sub>v</sub>∪{v}</sub> <i>a</i><sub>v,u</sub><b>W</b><sup>(l),Q</sup> ⊗ <b>h</b><sub>u</sub><sup>(l),Q</sup>), ∀ v ∈ V </p>

where we use the superscript <sup>Q</sup> to denote the Quaternion space; <i>a</i><sub>v,u</sub> is an edge constant between nodes v and u in the re-normalized adjacency matrix; <b>W</b><sup>(l),Q</sup> is a quaternion weight matrix; <b>h</b><sub>u</sub><sup>(0),Q</sup> is the quaternion feature vector of node v; and g can be a nonlinear activation function such as ReLU, and g is adopted to each quaternion element [12] as: g(<i>q</i>) = g(<i>q</i><sub>r</sub>) + g(<i>q</i><sub>i</sub>)<b>i</b> + g(<i>q</i><sub>j</sub>)<b>j</b> + g(<i>q</i><sub>k</sub>)<b>k</b>. 

We also represent the quaternion vector <b>h</b><sub>u</sub><sup>(l),Q</sup> ∈ H<sup>n</sup> and the quaternion weight matrix <b>W</b><sup>(l),Q</sup> ∈ H<sup>mxn</sup> as:

<p align="center">
	<img src="https://raw.githubusercontent.com/daiquocnguyen/daiquocnguyen.github.io/master/_pages/quaternion_vector_matrix.png" width="325">
</p>

We now express the Hamilton product ⊗ between <b>W</b><sup>(l),Q</sup> and <b>h</b><sub>u</sub><sup>(l),Q</sup> as:

<p align="center">
	<img src="https://raw.githubusercontent.com/daiquocnguyen/daiquocnguyen.github.io/master/_pages/matrix_vector_multiplication.png" width="385">
</p>

If we use any slight change in the input <b>h</b><sub>u</sub><sup>(l),Q</sup>, we get an entirely different output [16], leading to a different performance.
This phenomenon is one of the crucial reasons why the Quaternion space provides highly expressive computations through the Hamilton product compared to the Euclidean and complex vector spaces.
The phenomenon enforces the model to learn the potential relations within each hidden layer and between the different hidden layers, hence increasing the graph representation quality.

<b>QGNN for semi-supervised node classification.</b> We consider <b>h</b><sub>v</sub><sup>(L),Q</sup>, which is the quaternion vector representation of node v at the last <i>L</i>-th QGNN layer.
To predict the label of node v, we simply feed <b>h</b><sub>v</sub><sup>(L),Q</sup> to a prediction layer followed by a softmax layer as follows:

<p align="center"> &ycirc;<sub>v</sub> = softmax(∑<sub>u∈N<sub>v</sub>∪{v}</sub> <i>a</i><sub>v,u</sub><b>W</b> Vec(<b>h</b><sub>v</sub><sup>(L),Q</sup>)), ∀ v ∈ V </p>

where Vec(.) denotes a concatenation of the four components of the quaternion vector. For example, 

<p align="center"> Vec(<b>h</b><sub>v</sub><sup>(L),Q</sup>) = [<b>h</b><sub>v,r</sub><sup>(L)</sup>; <b>h</b><sub>v,i</sub><sup>(L)</sup>; <b>h</b><sub>v,j</sub><sup>(L)</sup>; <b>h</b><sub>v,k</sub><sup>(L)</sup>] </p>

<b>QGNN for graph classification.</b> We employ a concatenation over the vector representations <b>h</b><sub>v</sub><sup>(l),Q</sup> of node v at the different QGNN layers to construct the node embedding <b>e</b><sub>v</sub><sup>Q</sup>.
And then we use the sum pooling to obtain the embedding <b>e</b><sub>G</sub><sup>Q</sup> of the entire graph G as: 
<b>e</b><sub>G</sub><sup>Q</sup> = ∑<sub>∀v∈V</sub> <b>e</b><sub>v</sub><sup>Q</sup>.
To perform the graph classification task, we also use Vec(.) to vectorize <b>e</b><sub>G</sub><sup>Q</sup>, which is then fed to a single fully-connected layer followed by the softmax layer to predict the graph label.

## Conclusion<a name="conclusion"></a>

As this work represents a fundamental research problem in representing graphs, QGNN has demonstrated to be useful through experimental evaluations for downstream tasks of graph classification, node classification, and text classification.

Please cite [our paper](https://arxiv.org/abs/2008.05089) whenever QGNN is used to produce published results or incorporated into other software:

	@article{Nguyen2020QGNN,
		author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dinh Phung},
		title={Quaternion Graph Neural Networks},
		journal={NeurIPS 2020 Workshop on Differential Geometry meets Deep Learning. arXiv preprint arXiv:2008.05089},
		year={2020}
	}

## References

[1] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In Proceedings of the International Conference on Learning Representations (ICLR), 2017.

[2] William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Proceedings of the Advances in Neural Information Processing Systems, pages 1024–1034, 2017.

[3] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio.  Graph Attention Networks. In Proceedings of the International Conference on Learning Representations(ICLR), 2018.

[4] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural Message Passing for Quantum Chemistry.  In Proceedings of the International Conference on Machine Learning, pages 1263–1272, 2017.

[5] Muhan Zhang, Zhicheng Cui, Marion Neumann, and Yixin Chen. An End-to-End Deep Learning Architecture for Graph Classification. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018.

[6] Rex Ying,  Jiaxuan  You,  Christopher  Morris,  Xiang  Ren,  William  L.  Hamilton,  and  Jure Leskovec. Hierarchical graph representation learning with differentiable pooling. In Proceedings of the Advances in Neural Information Processing Systems, pages 4805–4815, 2018.

[7] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How Powerful Are Graph Neural Networks? In Proceedings of the International Conference on Learning Representations (ICLR), 2019.

[8] Cătălina Cangea, Petar Veličković, Nikola Jovanović, Thomas Kipf, and Pietro Liò. Towards sparse hierarchical graph classifiers. arXiv preprint arXiv:1811.01287, 2018.

[9] William Rowan Hamilton. Ii. on quaternions; or on a new system of imaginaries in algebra. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 25(163):10–13,1844.

[10] Xuanyu Zhu, Yi Xu, Hongteng Xu, and Changjian Chen.  Quaternion convolutional neural networks.  In Proceedings of the European Conference on Computer Vision (ECCV), pages631–647, 2018.

[11] Chase J Gaudet and Anthony S Maida. Deep quaternion networks. In Proceedings of the International Joint Conference on Neural Networks (IJCNN), pages 1–8, 2018.

[12] Titouan Parcollet, Mirco Ravanelli, Mohamed Morchid, Georges Linarès, Chiheb Trabelsi,Renato De Mori, and Yoshua Bengio. Quaternion recurrent neural networks. In Proceedings of the International Conference on Learning Representations (ICLR), 2019.

[13] Titouan Parcollet, Ying Zhang, Mohamed Morchid, Chiheb Trabelsi, Georges Linarès, RenatoDe  Mori,  and  Yoshua  Bengio.   Quaternion  convolutional  neural  networks for end-to-end automatic speech recognition. In Proceedings of the Interspeech, pages 22–26, 2018.

[14] Shuai Zhang, Yi Tay, Lina Yao, and Qi Liu.  Quaternion knowledge graph embeddings.  In Proceedings of the Advances in Neural Information Processing Systems, pages 2731–2741, 2019.

[15] Yi Tay, Aston Zhang, Anh Tuan Luu, Jinfeng Rao, Shuai Zhang, Shuohang Wang, Jie Fu, and Siu Cheung Hui. Lightweight and efficient neural natural language processing with quaternion networks.  In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1494–1503, 2019.

[16] Titouan Parcollet, Mohamed Morchid, and Georges Linarès.  A survey of quaternion neural networks. Artificial Intelligence Review, pages 1–26, 2019.
