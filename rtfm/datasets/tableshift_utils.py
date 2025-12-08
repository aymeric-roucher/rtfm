import json

try:
    from tableshift.core import PreprocessorConfig
    from tableshift.core.tabular_dataset import Dataset

    _TABLESHIFT_AVAILABLE = True
except ImportError:
    PreprocessorConfig = Dataset = object  # type: ignore
    _TABLESHIFT_AVAILABLE = False

from rtfm.arguments import (
    DataArguments,
    CAT_FEATURE_TRANSFORM_MAPPING,
    NUM_FEATURE_TRANSFORM_MAPPING,
)


def get_dataset_info(dset: Dataset) -> str:
    """Get the string representation of dataset info."""
    if not _TABLESHIFT_AVAILABLE:
        raise ImportError(
            "tableshift is not installed. Install tableshift to fetch raw datasets."
        )
    ds_info = dset._get_info()
    ds_info["task"] = dset.name
    return json.dumps(ds_info)


# For selected datasets, kbins preprocessing is used in the original tableshift
# data. In order to obtain the same underlying sample (or to avoid dropping
# too much data due to missing values when kbins is not used), we do *not*
# drop rows containing na values for these datasets. Otherwise, we use the
# standard/default preprocessing of TableShift.
# For dhs_diabetes, we keep na values since for some columns
# there are many NA values, and some columns are only defined for
# specific regions.
NO_DROPNA_EXPERIMENTS = [
    "physionet",
    "dhs_diabetes",
]


def fetch_preprocessor_config_from_data_args(
    data_args: DataArguments, experiment: str
) -> PreprocessorConfig:
    if not _TABLESHIFT_AVAILABLE:
        raise ImportError(
            "tableshift is not installed. Install tableshift to fetch raw datasets, "
            "or train with preserialized WebDataset shards."
        )
    numeric_value_handling = NUM_FEATURE_TRANSFORM_MAPPING[
        data_args.feature_value_handling
    ]
    categorical_value_handling = CAT_FEATURE_TRANSFORM_MAPPING[
        data_args.feature_value_handling
    ]
    return PreprocessorConfig(
        numeric_features=numeric_value_handling,
        categorical_features=categorical_value_handling,
        use_extended_names=data_args.feature_name_handling == "map",
        map_targets=data_args.targets_handling == "map",
        dropna=None if experiment in NO_DROPNA_EXPERIMENTS else data_args.dropna,
        cast_targets_to_default_type=False,
    )
