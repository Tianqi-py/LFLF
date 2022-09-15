# LFLF
Layerwise Feature Label Fusion Model Repository

Data:


the data folder with the datasets's name is the original dataset, containing its fetaures, labels and edges


the data folder with a number after the data's name (eg: Blogcatalog_n ) contains 3 splits of dataset using n% of data for training and ((1-n)/2)% for validation and ((1-n)/2)% for test.


In the paper, we use 60% as training data, 20% as traning data, 20% as test data.


full graph in pcg is in folder "pcg", the graph without isolated nodes is in "pcg_removed_isolated_nodes"


Hypersphere datasets are named after the rule "Hypersphere_rel_irre_red", i.e. the number of relevant features, irrelevant features and redundant features


Code:


models in : LFLF.py and LFLF_sup.py

the supervised and unsupervised models have variants:

LFLF_SAGE， LFLF_GCN, LFLF_GAT using the GraphSAGE, GCN, GAT convolutional layer correspondingly.


LFLF.py: unsupervised models
LFLF_sup.py: supervised models



utils: data processing functions


metrics: evaluation functions, loss,


main: for a quick run, change the parameter in main.py and run

main_sup: run models with supervised loss
main-unsup: run models with unsupervised loss



baselines:

2 layers GCN, GAT, GraphSAGE, MLP, H2GCN in supervised setting
