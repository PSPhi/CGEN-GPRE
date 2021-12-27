# [Molecular conditional generation and property analysis of non-fullerene acceptors with deep learning](https://www.mdpi.com/1422-0067/22/16/9099)
The molecular conditional generation and property prediction models are built with Pytorch (>=v1.7) and DGL-LifeSci.

## Structure
`utils.py` : Dataset preparation and utils function.

`model.py` : Generative and prediction model.

`config.py` : Parameters of the two models.

`cgen.py` : Code for training and testing the generative model.

`pre.py` : Code for training and testing the prediction model.

`sample.py` : Code for sampling.


## References
[1] Peng, S.-P.; Zhao, Y. Convolutional Neural Networks for the Design and Analysis of Non-Fullerene Acceptors. J. Chem. Inf. Model. 2019, 59, 4993–5001. [[Paper]](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00732) [[Code]](https://github.com/PSPhi/CNN-for-NFA)

[1] Gehring, J.; Auli, M.; Grangier, D.; Yarats, D.; Dauphin, Y. N. Convolutional Sequence to Sequence Learning. 2017. [[Paper]](http://arxiv.org/abs/1705.03122) [[Code]](https://github.com/pytorch/fairseq)

[2] Bai, S.; Kolter, J. Z.; Koltun, V. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. 2018. [[Paper]](http://arxiv.org/abs/1803.01271) [[Code]](https://github.com/locuslab/TCN)

[3] Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò, P.; Bengio, Y. Graph Attention Networks. 2018. [[Paper]](http://arxiv.org/abs/1710.10903)

[4] Lopez S A, Sanchez-Lengeling B, de Goes Soares J, et al. Design principles and top non-fullerene acceptor candidates for organic photovoltaics[J]. Joule, 2017, 1(4): 857-870. [[Paper]](https://www.sciencedirect.com/science/article/pii/S2542435117301307) [[Code]](https://github.com/couteiral/ORGANIC)
