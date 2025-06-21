Although the files for synthetic/original data have a `.pickle` extension, and the train/nontrain ones use `.pt`, all should get loaded directly with `pickle`.

There are two datasets sources, [twitch](https://snap.stanford.edu/data/twitch-social-networks.html) and [event](https://www.kaggle.com/competitions/event-recommendation-engine-challenge).

The files with the `original` suffix contain the single large, connected, and attributed graph from which subgraphs get sampled.

They have the format `tuple[pandas.DataFrame, networkx.Graph]`.

The other files contain a list of 256 entries each, with graphs sampled via random walks from the corresponding full graph.

They have the format `list[tuple[pandas.DataFrame, networkx.Graph]]`.

In all cases, indices in the DataFrame correspond to nodes in the graph.

Use `networkx` 3.2.1 and `pandas` 1.5.3 or other inter-compatible versions to load the data.

The `train`, `nontrain`, and `synthetic` represent the real samples used for training, real samples not used for training, and synthetic samples.

Given to the nature of the [employed sampling algorithm](https://github.com/Ashish7129/Graph_Sampling/blob/master/Graph_Sampling/SRW_RWF_ISRW.py#L88), there is a slight chance of ending up with more than one component in a subgraph, which is the case for a few samples in `event_train.pt`.

Furthermore, the training run to obtain the checkpoints was a bit short, so there is a small ratio (around 5% for twitch, and 12% for event) of graphs with more than one component in the synthetic samples.

The file `DLGrapher-dev.zip` contains the source code, while the files `twitch.ckpt` and `event.ckpt` are the checkpoints.

To run inferenceon the checkpoints, set as the working directory `src` from the source code, and use the command with:

`main.py model=hybrid dataset=DS +experiment=tablegraph_mia ++general.name=DS_mia ++general.test_only=<CKPT_PATH>`

replacing `DS` with `twitch` or `event` and `<CKPT_PATH>` with the location of the corresponding checkpoint file.

Appending `++general.final_model_samples_to_generate=N` to the command allows specifying a number `N` of samples to generate (the default is 256).
