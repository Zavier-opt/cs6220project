# CS 6220 Movie Recommendation System Based on the Peer to Peer Federated Learning

## Project Members:
Ziwei Cao, Chenyang Zhang, Junbai Tian, Linli Tang, Yinzhu Quan

## Project description:
Build a recommendation system based on Federated Learning 
to help users find movies that really interest them based on 
their preferences. Make experiments to detect the impact of 
unbalanced data distribution on the prediction of p2p FL model.
In our experiment, we simulate an environment which includes 20 nodes,
and consider four scenarios:
1. Split training data according to users. 
Each node has different number of users (the total number of nodes: 20); 
2. Split training data according to movies. 
Each node has different number of movies (the total number of nodes: 20); 
3. Split training data according to age of users. 
4. Split training data according to occupation of users.

## Data source:
The dataset we use is MovieLens 100K dataset. Data source: https://grouplens.org/datasets/movielens/100k/

## File structure:
- data : Include the pre-prepared datasets for each node, which are split by following the configurations in the report.
- models: Include the neural network models defined in the report.
- training_logs: Training logs of each node
- main_movie.py : The main function of the experiments.
- moviefederated_dm.py: Federated Learning function for the model1.
- moviefederated_dm2.py: Federated Learning function for the model2.
- utilits.py: Define the utilities used in the experiments.
- requirements.txt: List the requirements of this project.

## Execution method:
Run the main_movie.py file. Provide the MovieFederatedDM() function 
with different arguments to control the experiment.



