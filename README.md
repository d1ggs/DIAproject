# DIAproject

This repository contains the implementation of a system for combining influence maximization techniques and pricing to maximize revenues in a social network advertising scenario.

## Assignment
 
The goal is modeling a scenario in which a seller is pricing some products and spends a given budget on social networks to persuade more and more nodes to buy the products, thus artificially increasing the demand. The seller needs to learn both some information on the social networks and the conversion rate curves.
  1. Imagine:
     * three products to sell, each with an infinite number of units, in a time horizon T;
     * three social networks composed of thousands of nodes, such that each social network is used to sell a different product;
     * the activation probabilities of the edges of the social networks are linear functions in the values of the features (>3), potentially different in the three social networks;
     * three seasonal phases such that the transitions from a phase to the subsequent one are abrupt;
     * a conversion rate curve for each social network and each phase, returning the probability that a generic node of the social network buys a product (notice that the phases affect the conversion rate curve, but not the activation probabilities of the social networks).
  2. Design an algorithm maximizing the social influence in every single social network once a budget, for that specific social network, is given. Plot the approximation error as the parameters of the algorithms vary for every specific network.
  3. Design a greedy algorithm such that, given a cumulative budget to perform jointly social influence in the three social networks, finds the best allocation of the budget over the three social networks to maximize the cumulative social influence. Plot the approximation error as the parameters of the algorithms vary for every specific network.
  4. Apply a combinatorial bandit algorithm to the situation in which the activation probabilities are not known and we can observe the activation of the edges. Plot the cumulative regret as time increases. 
  5. Design a learning pricing algorithm to maximize the cumulative revenue and apply it, together with the algorithm to make social influence, to the case in which the activation probabilities are known. In doing that, simplify the environment adopting a unique seasonal phase for the whole time horizon. The daily number of customers interested to buy each product is the number of nodes of the corresponding social network activated by social influence. For simplicity, imagine that every day the seller makes social influence to convince the nodes to buy the products and the activated nodes are the users that will try to buy the product. The actual purchase depends on the price charged by the seller and the conversion rate curve. For simplicity, assume that a node that has bought a product in a day can buy it also the subsequent days if activated by social influence. Plot the cumulative regret.
  6. Design a learning pricing algorithm to maximize the cumulative revenue when there are seasonal phases and apply it, together with the algorithm to make social influence, to the case in which the activation probabilities are known. The number of customers interested to buy each product is the number of nodes of the corresponding social network activated by social influence, as in the previous step. Plot the cumulative regret.
  7. Plot the cumulative regret in the case the seller needs to learn both the activation probabilities and conversion rate curves simultaneously.

## Solution
For details on the assumptions and algorithms used to solve each assignment, have a look at our [presentation](Pricing+Influence.pdf).
<!-- ## Social influence maximization

## Pricing -->

## Requirements
To install requirements after cloning the repo:
```
pip install -r requirements.txt
```

## Running the code
There are no command line arguments, the code runs on Python 3.
Each python file in the main folder represents the corresponding assignment item, for instance to run assignment 4: 
```
python ex4.py
```

## Authors
Matteo Carretta, Lorenzo Casalini, Federica Ilaria Cesti, Diego Piccinotti, Umberto Pietroni
