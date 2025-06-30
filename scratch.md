Needs inspecting:
1. How do I generate compelete synthetic graph data from original data, not just samples.
   - How do i get access to `https://gitlab.ewi.tudelft.nl/dmls/research/graph-synthesizing/nets-eval-common.git` to run DLGrapher? Running this is crucial for experimentation anyway.
   - `python main.py model=hybrid dataset=twitch +experiment=tablegraph_mia ++general.name=twitch_mia ++general.test_only=../checkpoints/twitch/twitch.ckpt ++general.final_model_samples_to_generate=1` could work.
1. I could maybe perform TSTS setting using the available sample subgraphs but performing TSTF will be difficult. It might be hard to modify `get_inductive_split()` and ensure same properties required for efficient split for target and shadow model.  