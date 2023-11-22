# ChainFL Documentaion
## Blockchain for FL
1. Record model ownership information (based on model watermark) on Blockchain.
2. Design Smart Contract to record and supervise training information.

## Introduction
ChainFL is a simple federated learning framework embedded with blockchain(Etherenum Network). In this framework, the blockchain acts as a FL training manager to record training message and supervise the entire network. While being able to act the normal federated training experiment, the framework also provide a real blockchain environment to promote the research on blockchain empowered federated learning.

## Dependence

Ethereum Environment 
- [Nodejs](https://nodejs.org/en)

Node.js >= 16.0.0 and npm >= 7.10.0
- Ganache
```
npm install ganache --global
```
Python Environment
- Anaconda 

1. Python: 3.6 ~ 3.9

2. PyTorch: 1.13
- Brownie
```
pip install eth-brownie
```

## Code Structure and Usage

- Quick Start
```
python test.py
```

```
#test.py
import logging
from task import Task
import config.benchmark

global_args, train_args = config.benchmark.FashionMNIST().get_args()

if __name__=="__main__":
    classification_task = Task(global_args=global_args, train_args=train_args)
    classification_task.run()
```


- Add Customize task
1. Create benchmark (basic parameters) in benchmark.py
2. Write training logic in trainer
3. Create model align with ModelFactory

- Add New FL Algorithm
1. New Aggregator (Server)
2. New Trainier   (Client)


## Implemented Algorithm
- Federated Learning Algorithm

- Client Selection Strategy on Blochchain

- Model Ownership Verification (Tokenized Model) on Blockchain