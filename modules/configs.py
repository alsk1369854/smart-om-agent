from . import types

BGL_WORK_CONFIG = types.WorkConfig(
    name="BGL",
    system_name="BlueGene/L supercomputer system",
    dataset_config=types.DatasetConfig(
        path="./data/BGL/BGL.log",
        fromat="<label> <timestamp> <date> <node> <time> <node_repeat> <type> <component> <level> <content>",
        start_line=0,
        end_line=None,
        timestamp_column="timestamp",
        feat_columns=["level", "content"],
        label_column="label",
    ),
)

LIBERTY_WORK_CONFIG = types.WorkConfig(
    name="Liberty",
    system_name="Liberty supercomputer system",
    dataset_config=types.DatasetConfig(
        path="./data/Liberty/Liberty.log",
        fromat="<label> <timestamp> <date> <user> <month> <day> <time> <location> <content>",
        start_line=40000000,
        end_line=45000000,
        timestamp_column="timestamp",
        feat_columns=["content"],
        label_column="label",
    ),
)


THUNDERBIRD_WORK_CONFIG = types.WorkConfig(
    name="Thunderbird",
    system_name="Thunderbird supercomputer system",
    dataset_config=types.DatasetConfig(
        path="./data/Thunderbird/Thunderbird.log",
        fromat="<label> <timestamp> <date> <user> <month> <day> <time> <location> <content>",
        start_line=160000000,
        end_line=170000000,
        timestamp_column="timestamp",
        feat_columns=["content"],
        label_column="label",
    ),
)

TEST_WORK_CONFIG = types.WorkConfig(
    name="TEST", # based on BGL
    system_name="BlueGene/L supercomputer system",
    dataset_config=types.DatasetConfig(
        path="./data/BGL/BGL.log",
        fromat="<label> <timestamp> <date> <node> <time> <node_repeat> <type> <component> <level> <content>",
        start_line=30000,
        end_line=31000,
        timestamp_column="timestamp",
        feat_columns=["level", "content"],
        label_column="label",
    ),
)

WORK_CONFIG_MAP: dict[types.DatasetTypes, types.WorkConfig] = {
    "BGL": BGL_WORK_CONFIG,
    "Liberty": LIBERTY_WORK_CONFIG,
    "Thunderbird": THUNDERBIRD_WORK_CONFIG,
    "test": TEST_WORK_CONFIG,
}
