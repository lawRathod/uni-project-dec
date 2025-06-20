from analysis.spectre_utils import SpectreSamplingMetrics

class NetSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre'])
