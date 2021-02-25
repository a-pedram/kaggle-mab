# kaggle-mab
A collection of different agents I wrote for a kaggle competition(on multi armed bandit) and a few publicly published ones.

<a href="https://www.kaggle.com/c/santa-2020"> The competition page </a>


The "agents" folder contains all agents used I this competition both for the purpose of local evaluation and competing in the contest. To start you can run "test.py". This file includes the code required to simulate environment of the competition. To compare an agent in the "agents" with a few others in that folder, you only need to change the following code:
<pre>>
agent1 =  ["agents.ag_newHero",{}]
agents = [
     ["agents.ag_followHim",{}],
     ["agents.ag_followNFool",{}],
     ["agents.ag_followHimHisScore",{}],
]
</pre>

The best agent on the competition's leader-board was "agents/ag_newHero.py". I have explained its mechanism  <a href="https://www.kaggle.com/c/santa-2020/discussion/217537"> in this kaggle post. </a>