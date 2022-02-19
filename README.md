# committor_learning_schnet

This is a PyTorch script that adapts the previous CPU focused package, [tps-torch](https://github.com/muhammadhasyim/tps-torch), into a lean, mean, GPU-running machine that works as long as the user provides the necessary configurations and free energy data.
See our original paper for more details on the original scheme:
- M. R. Hasyim, C. H. Batton, K. K. Mandadapu, "Supervised Learning and the Finite-Temperature String Method for Computing Committor Functions and Reaction Rates", [arXiv:2107.13522](https://arxiv.org/abs/2107.13522) (2021)

The script does committor learning using a modified form of the SchNet implementation available with PyTorch Geometric (`schnet_committor.py`).
An initial estimate of the committor profile can be generated using `train_init.py`.
The main training is then done using `train_committor.py` and `train_committor_r.py`, with the script split into a restartable portion in order to be more amendable to some computing resources.
The scripts are useless without the necessary configurations and other data needed to implement the reweighting schemes to obtain a good estimate of the BKE loss, but hopefully serves as a good framework for future work that wishes to adapt our work on already available data.
