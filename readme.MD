

# Chainlet Orbits: Topological Address Embedding for the Bitcoin Blockchain

 To address the computational and interpretability limitations of the existing techniques for large Bitcoin transaction graphs, we define an efficient topological node embedding technique suitable for Bitcoin transaction addresses. Using our novel node embedding techniques, we study e-crime in Bitcoin networks by focusing on special substructures induced by the illicit behavior and show that using our node embedding technique, we can efficiently match the existing illicit activity detection accuracy of the graph neural networks while allowing interpretable and explainable machine learning models.

 
## Bitcoin Transaction Network Data and Address Label Datasets

The src/orbit code parses the Bitcoin transaction network and adds address labels to the data. The project utilizes a 24-hour time window centered on the Central Standard Time zone to extract daily transactions on the network and construct the Bitcoin graph. The address orbits are used without any normalization or preprocessing, as extracting orbits is not demanding in terms of computational resources.


![Bitcoin Transaction Graph](https://user-images.githubusercontent.com/124116680/225409110-c508fffc-0946-4b7d-8dd6-2deb81f4546a.png)


For the graph shown in the figure, we identify the orbit type of each address in the graph:

Use the OrbitAnalyzer.py file is used to get the chainlet orbit of each address node in a given graph.


## Address Classification 
We report the average accuracy of five runs, along with their standard deviation. Training and test split is 80/20. 
In experiments on the BitcoinHeist data, we use binary classification (white/ransomware) and use a Random Forest with 300 trees.

In experiments on the full orbit data, we use a one-versus-rest Random Forest classifier.

* Graph Neural Networks
  We use GCN, GIN, GraphSage and DGCNN in our experiments. We used 1000 epochs. For hyperparemeters, please see the config.yml files under src/GNN.

* Orbit-based Classifiers
  With orbits, we use OneVsRestClassifier of RandomForestClassifier of the Scikit library.
  The default parameters of the `sklearn.ensemble.RandomForestClassifier` are as follows:

```python
RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
)
```
 
| Parameters | Value|
|----------|----------|
| Model | OneVsRestClassifier (RandomForestClassifier)|
| Sample size | 5000,100000,500000,1000000 |
| Test size | 0.2|
| Number of ransomware families in the dataset | 27|
| Unique ransomware addresses | 19930|
| Unique darknet addresses | 7557 |
| Hardware Experimental Setup | Dell PowerEdge R630, featuring an Intel Xeon E5-2650 v3 Processor (10-cores, 2.30 GHz, 20MB Cache), and 192GB of RAM (DDR4-2133MHz) |
