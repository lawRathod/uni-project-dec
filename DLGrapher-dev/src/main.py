import rdkit as _rdkit
try:
    import graph_tool as _gt
except ModuleNotFoundError:
    print("Graph tool not found")
import os
import pathlib
import warnings
from typing import cast

import nets_eval_common
import torch
torch.cuda.empty_cache()
import torch._dynamo.config as dynamo_config
import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from torch_geometric.data import Data, Dataset
from lightning_fabric.utilities.warnings import PossibleUserWarning

import utils
from metrics.abstract_metrics import TrainAbstractMetrics
from tab_ddpm.tab_ddpm import TabDDPM

from compress import FeatsCompress, FullCompress, StructCompress
from datasets.abstract_dataset import AbstractTableGraphDataset
from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion_model_hybrid import HybridDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures


warnings.filterwarnings("ignore", category=PossibleUserWarning)
torch.set_float32_matmul_precision("high")


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    final_model_samples_to_generate = cfg.general.final_model_samples_to_generate
    final_model_samples_to_save = cfg.general.final_model_samples_to_save
    final_model_chains_to_save = cfg.general.final_model_chains_to_save
    model_kwargs.pop('compress', None)
    model_kwargs.pop('tab_gen', None)
    wandb_ = cfg.general.wandb
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    elif cfg.model.type == 'hybrid':
        model = HybridDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    if hasattr(model, 'compress'):
        model_kwargs['compress'] = model.compress
    if hasattr(model, 'tab_gen'):
        model_kwargs['tab_gen'] = model.tab_gen
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.general.final_model_samples_to_generate = final_model_samples_to_generate
    cfg.general.final_model_samples_to_save = final_model_samples_to_save
    cfg.general.final_model_chains_to_save = final_model_chains_to_save
    cfg.general.wandb = wandb_
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    model_kwargs.pop('compress', None)
    model_kwargs.pop('tab_gen', None)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    elif cfg.model.type == 'hybrid':
        model = HybridDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)

    if hasattr(model, 'compress'):
        model_kwargs['compress'] = model.compress
    if hasattr(model, 'tab_gen'):
        model_kwargs['tab_gen'] = model.tab_gen

    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    accelerator = 'gpu' if use_gpu else 'cpu'

    if dataset_config["name"] in ['sbm', 'comm20', 'planar']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(cfg, None)
        if dataset_config['name'] == 'sbm':
            sm_class = SBMSamplingMetrics
        elif dataset_config['name'] == 'comm20':
            sm_class = Comm20SamplingMetrics
        else:
            sm_class = PlanarSamplingMetrics

        sampling_metrics = sm_class(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule)
        train_metrics = TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization(cfg.train.seed)

        if cfg.general.compress is None:
            compress = None
        elif cfg.general.compress == 'struct':
            compress = StructCompress()
        else:
            raise ValueError("The specified compression type is invalid for unattributed data.")

        if compress:
            compress.prepare(datamodule)
            datamodule = SpectreGraphDataModule(cfg, compress)
            sampling_metrics = sm_class(datamodule)
            dataset_infos = SpectreDatasetInfos(datamodule)

        if cfg.model.type != 'continuous' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from src.metrics.molecular_metrics_hybrid import TrainMolecularMetricsHybrid
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset, guacamol_dataset, moses_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = guacamol_dataset.get_train_smiles(
                cfg=cfg, datamodule=datamodule,
                dataset_infos=dataset_infos, evaluate_dataset=False)

        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = moses_dataset.get_train_smiles(
                cfg=cfg, datamodule=datamodule,
                dataset_infos=dataset_infos, evaluate_dataset=False)
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type != 'continuous' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        elif cfg.model.type == 'hybrid':
            train_metrics = TrainMolecularMetricsHybrid(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles, datamodule)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in nets_eval_common.dataset_names():
        from datasets.net_dataset import NetDataModule, NetDatasetInfos
        from analysis.net_utils import NetSamplingMetrics
        # from pytorch_lightning import seed_everything

        # seed_everything(0)
        datamodule = NetDataModule(cfg, None)
        sampling_metrics = NetSamplingMetrics(datamodule)

        dataset_infos = NetDatasetInfos(datamodule)
        train_metrics = TrainAbstractMetrics()
        visualization_tools = None

        if cfg.general.compress is None:
            compress = None
        elif cfg.general.compress == 'struct':
            compress = StructCompress()
        else:
            raise ValueError("The specified compression type is invalid for unattributed data.")

        if compress:
            compress.prepare(datamodule)
            datamodule = NetDataModule(cfg, compress)
            sampling_metrics = NetSamplingMetrics(datamodule)
            dataset_infos = NetDatasetInfos(datamodule)

        if cfg.model.type != 'continuous' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features,
                        'compress': compress}

    elif dataset_config["name"] == "karateclub":
        from analysis.karateclub_utils import KarateClubSamplingMetrics
        from analysis.visualization import NonMolecularNxVisualization
        from datasets.karateclub import (KarateClubDataModule,
                                         KarateClubDatasetInfos)

        datamodule = KarateClubDataModule(cfg)
        sampling_metrics = KarateClubSamplingMetrics(datamodule)

        dataset_infos = KarateClubDatasetInfos(datamodule)
        train_metrics = TrainAbstractMetrics()
        visualization_tools = NonMolecularNxVisualization(cfg.dataset.name)

        if cfg.model.type != 'continuous' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] == "ego-twitter":
        from analysis.net_utils import NetSamplingMetrics
        from analysis.visualization import NonMolecularVisualization
        from datasets.ego_dataset import EgoDataModule
        from datasets.net_dataset import NetDatasetInfos

        datamodule = EgoDataModule(cfg)
        sampling_metrics = NetSamplingMetrics(datamodule)
        dataset_infos = NetDatasetInfos(datamodule)
        train_metrics = TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization(cfg.train.seed)

        if cfg.model.type != 'continuous' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['twitch', 'event', 'ogbn-arxiv']:
        from analysis.spectre_utils import SpectreSamplingMetrics
        from analysis.table_graph_utils import (TableGraphDatasetInfos,
                                                TableGraphSamplingMetrics,
                                                table_graph_path_to_df)
        from analysis.visualization import (NonMolecularVisualization,
                                            TableGraphCompressedVisualization,
                                            TableGraphVisualization)
        from datasets.event_dataset import EventDataModule
        from datasets.ogbnarxiv_dataset import OgbnArxivDataModule
        from datasets.twitch_dataset import TwitchDataModule

        ds_name = dataset_config["name"]

        if cfg.general.hd_src == 'self' and cfg.model.type != 'hybrid':
            raise ValueError("Joint feature generation is only supported by hybrid model")

        if cfg.general.hd_src not in {'self', 'none'} and cfg.general.compress in {'feats', 'full'}:
            raise ValueError("Cannot apply compression to node features from auxiliary model")

        if ds_name == 'twitch':
            dm_class = TwitchDataModule
        elif ds_name == 'event':
            dm_class = EventDataModule
        elif ds_name == 'ogbn-arxiv':
            dm_class = OgbnArxivDataModule
        else:
            raise ValueError("Dataset not implemented")

        datamodule = dm_class(cfg, None, cfg.general.hd_src != 'none')
        dataset_infos = TableGraphDatasetInfos(datamodule)
        train_metrics = TrainAbstractMetrics()

        if cfg.general.hd_src == 'none':
            sampling_metrics = SpectreSamplingMetrics(
                datamodule=datamodule,
                compute_emd=False,
                metrics_list=['degree', 'clustering', 'orbit', 'spectre'])
            visualization_tools = None
        else:
            sampling_metrics = TableGraphSamplingMetrics(datamodule, dataset_infos)

            if cfg.general.compress and cfg.model.type=='hybrid':
                visualization_tools = TableGraphCompressedVisualization(
                    dataset_infos, cfg.train.seed)
            else:
                visualization_tools = TableGraphVisualization(
                    dataset_infos, cfg.train.seed)

        if  cfg.general.compress in {'feats', 'full'} and cfg.general.hd_src == 'none':
            raise ValueError('Can not perform feature compression on data without node features.')

        test_or_resume: bool = cfg.general.test_only or cfg.general.resume

        if cfg.general.compress is None:
            compress = None
        elif cfg.general.compress == 'struct':
            compress = StructCompress()
        elif cfg.general.compress == 'feats':
            compress = FeatsCompress(
                dataset_infos.num_classes, dataset_infos.cat_indices, accelerator)
        elif cfg.general.compress == 'full':
            compress = FullCompress(
                dataset_infos.num_classes, dataset_infos.cat_indices, accelerator)
        else:
            raise ValueError("The specified compression type is invalid.")

        if compress and not test_or_resume:
            compress.prepare(datamodule)

        ds = cast(AbstractTableGraphDataset, datamodule.train_dataset)

        if cfg.general.hd_src in {'none', 'self'} or test_or_resume:
            tab_gen = None
        elif cfg.general.hd_src == 'tvae':
            train_data = table_graph_path_to_df(ds.processed_paths[ds.file_idx["og_train"]])
            train_data = train_data.astype(
                {dt: 'object' for dt in train_data.select_dtypes(include='category').columns})
            st_md = SingleTableMetadata()
            st_md.detect_from_dataframe(train_data)
            tab_gen = TVAESynthesizer(st_md, epochs=1_000)
            tab_gen.fit(train_data)
        elif cfg.general.hd_src == 'tabddpm':
            train_data = table_graph_path_to_df(ds.processed_paths[ds.file_idx["og_train"]])
            val_data = table_graph_path_to_df(ds.processed_paths[ds.file_idx["og_val"]])
            test_data = table_graph_path_to_df(ds.processed_paths[ds.file_idx["og_test"]])
            tab_gen = TabDDPM(dataset_infos, cfg.dataset.tgt_col, epochs=10_000)
            tab_gen.fit(train_data, val_data, test_data, cfg.train.seed)
        else:
            raise ValueError("The specified tabular synthesizer is invalid.")

        datamodule = dm_class(cfg, compress, cfg.general.hd_src == 'self')
        dataset_infos = TableGraphDatasetInfos(datamodule)

        if cfg.model.type != 'continuous' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features,
                        'compress': compress, 'tab_gen': tab_gen}

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    train_ds_lens = cast(list[int], [cast(Data, d).num_nodes for d in datamodule.train_dataset])
    print(
        "#Train examples:",     len(datamodule.train_dataset), "|",
        "#Val examples:",       len(cast(Dataset, datamodule.val_dataset)), "|",
        "#Test examples:",      len(cast(Dataset, datamodule.test_dataset)), "|",
        "Min train nodes:",     min(train_ds_lens), "|",
        "Max train nodes:",     max(train_ds_lens), "|",
        "Mean train nodes:",    sum(train_ds_lens) / len(train_ds_lens)
    )

    if cfg.general.downstream:
        import downstream

        tti = cast(TableGraphDatasetInfos, dataset_infos).tti

        downstream.node_classification(
            ds.processed_paths[ds.file_idx['og_train']],
            cfg.general.downstream,
            ds.processed_paths[ds.file_idx['og_test']],
            ds.raw_paths[0], tti, cfg.dataset.tgt_col, accelerator)

        return

    if cfg.general.test_only or cfg.general.resume is not None:
        import torch.distributed as dist
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('gloo', rank=0, world_size=1)

        if cfg.general.test_only:
            # When testing, previous configuration is fully loaded
            cfg, _ = get_resume(cfg, model_kwargs)
            os.chdir(cfg.general.test_only.split('checkpoints')[0])
        else:
            # When resuming, we can override some parts of previous configuration
            cfg, _ = get_resume_adaptive(cfg, model_kwargs)
            os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    elif cfg.model.type == 'hybrid':
        model = HybridDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model_kwargs.pop('compress', None)
        model_kwargs.pop('tab_gen', None)
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    if cfg.general.compile:
        dynamo_config.suppress_errors = True
        model = cast(LightningModule, torch.compile(model))

    callbacks: list[Callback] = []

    if cfg.train.early_stop_patience:
        callbacks.append(EarlyStopping(
            monitor='val/epoch_loss', patience=cfg.train.early_stop_patience, strict=True))

    if cfg.train.save_model:
        # only runs on epochs divisible by both every_n_epochs & trainer.check_val_every_n_epoch
        save_top_k = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",  filename='{epoch}',
            monitor='val/epoch_NLL', save_top_k=5, mode='min', every_n_epochs=1)
        # runs on all epochs
        save_last = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1,
            save_on_train_epoch_end=True)
        callbacks.extend((save_top_k, save_last))

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    print(cfg)

    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator=accelerator,
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger=[])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(datamodule=datamodule, ckpt_path='last')
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
