# Less is More: SlimG for Accurate, Robust, and Interpretable Graph Mining

------------

Yoo, J.\*, Lee, M. C.\*, Shekhar, S., & Faloutsos, C. (2023, August). Less is More: SlimG for Accurate, Robust, and Interpretable Graph Mining. *29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 2023.

https://dl.acm.org/doi/10.1145/3580305.3599413

Please cite the paper as:

    @inproceedings{yoo2023less,
      title={Less is More: SlimG for Accurate, Robust, and Interpretable Graph Mining},
      author={Yoo, Jaemin and Lee, Meng-Chieh and Shekhar, Shubhranshu and Faloutsos, Christos},
      booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
      pages={3128--3139},
      year={2023}
    }

## Introduction

How can we solve semi-supervised node classification in various types of graphs possibly with noisy features and structures?

We propose SlimG, which exhibits the folloiwing desirable properties:
- C1.1 - **Accurate**, on both real-world and synthetic datasets, almost always winning or tying in the first place.
- C1.2 - **Robust**, being able to handle numerous real settings such as homophily, heterophily, no network effects, useless features.
- C1.3 - **Fast and Scalable**, using carefully chosen features, it takes only 32 seconds on million-scale real-world graphs (ogbn-Products) on a stock server.
- C1.4 - **Interpretable**, learning the largest weights on informative features, ignoring noisy ones, based on the linear decision function.

We also explain the reasons for its success, thanks to our three additional contributions:
- C2 - **Explanation**: We propose GNNExp, a framework for the systematic linearization of a GNN.
- C3 - **Sanity Checks**: We propose seven possible scenarios of graphs (homophily, heterophily, no network effects, etc.), which reveal the strong and weak points of each GNN.
- C4 - **Experiments**: We conduct extensive experiments to better understand the success of SlimG even with its simplicity.

## Experiments

![image](https://github.com/mengchillee/SlimG/assets/14501754/de0063cf-7bcb-4a78-92ba-683ff56aa8a3)

![image](https://github.com/mengchillee/SlimG/assets/14501754/569271b9-a574-452c-844c-5392be075dae)


## Usage
The code is written in Python 3.10.8 and built on a number of packages.

Please see "requirements.txt" for the package details.

To run the code, you can simply use the following comment:
`python src/main.py`

The datasets will be automatically downloaded when running for the first time.
