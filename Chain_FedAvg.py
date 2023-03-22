#!/usr/bin/env python
# coding: utf-8

# 注意，凡是代码中包含
# ```python
# np.random.rand(10,1)
# ```
# 的字眼都代表这是暂定的模型的数据结构，我在这里为了方便只是假设它是一个长度为10的向量，到实际操作的时候可以换成别的

# In[63]:


from hashlib import sha256
import numpy as np
import pandas as pd
import random
import json
import time


# In[64]:


#先构建区块的类结构Block
class Block:
    def __init__(self, index, transactions, timestamp, previous_hash):
        """
        Constructor for the `Block` class.
        :param index:         Unique ID of the block.   区块的序号
        :param transactions:  List of transactions.     区块内包含的交易记录（数据记录）
        :param timestamp:     Time of generation of the block.     区块生成时间
        :param previous_hash: Hash of the previous block in the chain which this block is part of.   前一个区块的哈希                                     
        """
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash # Adding the previous hash field

    def compute_hash(self):      #用于计算区块的哈希
        """
        Returns the hash of the block instance by first converting it
        into JSON string.
        """
        block_string = json.dumps(self.__dict__, sort_keys=True) # The string equivalent also considers the previous_hash field now
        return sha256(block_string.encode()).hexdigest()


# In[65]:


#构建区块链的类结构Blockchain
class Blockchain:
    # difficulty of PoW algorithm
    difficulty = 2

    
    def __init__(self):
        """
        Constructor for the `Blockchain` class.
        """
        self.unconfirmed_transactions = [] # data yet to get into blockchain
        self.chain = []
        self.create_genesis_block()
        
    def create_genesis_block(self):    #生成初始区块
        """
        A function to generate genesis block and appends it to
        the chain. The block has index 0, previous_hash as 0, and
        a valid hash.
        """
        genesis_block = Block(0, [], time.time(), "0")
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)

    @property
    def last_block(self):          #返回链中最新的区块
        """
        A quick pythonic way to retrieve the most recent block in the chain. Note that
        the chain will always consist of at least one block (i.e., genesis block)
        """
        return self.chain[-1]
    
    
    def proof_of_work(self, block):   #计算出符合要求的哈希的nonce值
        """
        Function that tries different values of the nonce to get a hash
        that satisfies our difficulty criteria.
        """
        block.nonce = 0

        computed_hash = block.compute_hash()
        while not computed_hash.startswith('0' * Blockchain.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()

        return computed_hash
    
    def add_block(self, block, proof):  #将区块加入到链中
        """
        A function that adds the block to the chain after verification.
        Verification includes:
        * Checking if the proof is valid.
        * The previous_hash referred in the block and the hash of a latest block
          in the chain match.
        """
        previous_hash = self.last_block.hash

        if previous_hash != block.previous_hash:
            return False

        if not Blockchain.is_valid_proof(block,proof):
            return False

        block.hash = proof
        self.chain.append(block)
        return True

    @classmethod
    def is_valid_proof(self, block, block_hash):        #检查哈希地址是否符合要求
        """
        Check if block_hash is valid hash of block and satisfies
        the difficulty criteria.
        """
        return (block_hash.startswith('0' * Blockchain.difficulty) and
                block_hash == block.compute_hash())
    
    def add_new_transaction(self, transaction):        #将新的交易（数据）加入到未确认的交易中
        self.unconfirmed_transactions.append(transaction)

    def mine(self):        #挖矿，生成新的区块并将unconfirmed_transaction加入其中，然后放置链上
        """
        This function serves as an interface to add the pending
        transactions to the blockchain by adding them to the block
        and figuring out proof of work.
        """
        if not self.unconfirmed_transactions:
            return False

        last_block = self.last_block

        new_block = Block(index=last_block.index + 1,
                          transactions=self.unconfirmed_transactions,
                          timestamp=time.time(),
                          previous_hash=last_block.hash)

        proof = self.proof_of_work(new_block)
        self.add_block(new_block, proof)
        self.unconfirmed_transactions = []
        return new_block.index


# In[66]:


#构建用户的类结构client
class client:
    
    def __init__(self):
        self.local_transaction = np.random.rand(10,1)     #待上传的本地模型
        self.global_transaction = np.random.rand(10,1)
        self.Blockchain = Blockchain()                    #用户本地自己的链
        
    def upload(self):
        self.Blockchain.add_new_transaction(self.local_transaction)


# In[67]:


#将用户们的数据放入到local_transaction中
def put_data_to_local(client_list,parameters):
    for i in range(len(client_list)):
        client_list[i].local_transaction = parameters[i]

#然后用户们将自己的local_transaction数据放入到Blockchain类的unconfirmed_transactions中，并通知所有客户
def upload_and_broadcast(client_list):
    for i in range(len(client_list)):
        client_list[i].upload()
        for j in range(len(client_list)):        #通知播报全体客户
            client_list[j].Blockchain.unconfirmed_transactions = client_list[i].Blockchain.unconfirmed_transactions
            
#随机选择一名用户并挖矿生成新的区块
def mine_and_broadcast(client_list):
    random_client = random.sample(client_list,1)
    random_client[0].Blockchain.mine()
    for i in range(len(client_list)):
        client_list[i].Blockchain = random_client[0].Blockchain


# ### 开始

# In[68]:


client_list = [client() for i in range(100)]        #创建包含100个用户的名单，当使用client这个类的时候就已经创建了初始区块了
print(client_list[6].Blockchain.chain[0].hash)      #展示一下其中一个客户


# In[69]:


#这里先假设一个模型的参数是一个长度为10的向量，总共有100个用户，其数据用这个parameters来表示
parameters = np.random.rand(100,10).tolist()


# In[82]:


#单个通信阶段的FedAvg，每个用户陆续上传数据到链中，然后将其包装成新的区块，
def single_FedAvg(parameters,client_list):
    put_data_to_local(client_list,parameters)
    upload_and_broadcast(client_list)
    mine_and_broadcast(client_list)
    global_model = np.array(client_list[99].Blockchain.last_block.transactions).mean(axis=0)
    for i in range(len(client_list)): 
        client_list[i].global_transaction = global_model

