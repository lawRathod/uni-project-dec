# DLGrapher: Dual Latent Diffusion For Attributed Graph Generation

Code is based upon that of [DiGress](https://github.com/cvignac/DiGress).

## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometric 2.3.1

  - Download anaconda/miniconda/conda-forge if needed

  - Create a complete environment (which also contains rdkit):

    ```conda env create -p .env -f environment.yml```

    or

    ```
    conda create -c conda-forge -p .env rdkit=2023.03.2 graph-tool=2.45 python=3.9
    pip install -r requirements.txt
    ```

  - Activate it:

    ```conda activate .env/```

  - Check that this line does not return an error:

    ```python3 -c 'from rdkit import Chem'```

  - Check that this line does not return an error:

    ```python3 -c 'import graph_tool as gt'```

  - Navigate to the `./src/analysis/orca` directory and compile `orca.cpp`:

     ```g++ -O2 -std=c++11 -o orca orca.cpp```

## Run the code

  - All code is currently launched through `python3 main.py` (with `src` as the working dir). Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 main.py +experiment=debug.yaml`. We advise to try to run the debug mode first
    before launching full experiments.
  - To run a code on only a few batches: `python3 main.py general.name=test`.
  - To run the continuous model: `python3 main.py model=continuous`
  - To run the discrete model: `python3 main.py`
  - You can specify the dataset with `python3 main.py dataset=guacamol`. Look at `configs/dataset` for the list
of datasets that are currently available

## Troubleshooting

`PermissionError: [Errno 13] Permission denied: '/[...]/src/analysis/orca/orca'`: You probably did not compile orca.
