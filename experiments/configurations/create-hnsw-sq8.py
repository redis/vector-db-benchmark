import json
import os

data_types = ["FLOAT32", "FLOAT16"]
hnsw_m = 32
ef_construction = 200
sq8_training_thresholds = {"sq8-zero": 0, "sq8-trained": 10240}
svs_compression = "LVQ8"
clients = [1, 8, 100]
breadths = [100, 200, 400, 800, 1600]
calibration_targets = [0.95, 0.99]
comparison_topKs = [10, 100]

out_dir = os.path.dirname(os.path.abspath(__file__))


def hnsw_collection(data_type, threshold=None):
    hnsw_config = {"M": hnsw_m, "EF_CONSTRUCTION": ef_construction}
    if threshold is not None:
        hnsw_config["COMPRESSION"] = "SQ8"
        hnsw_config["TRAINING_THRESHOLD"] = threshold
    return {"algorithm": "hnsw", "data_type": data_type, "hnsw_config": hnsw_config}


def svs_collection(data_type):
    svs_config = {
        "GRAPH_MAX_DEGREE": hnsw_m,
        "CONSTRUCTION_WINDOW_SIZE": ef_construction,
        "compression": svs_compression,
    }
    return {"algorithm": "svs-vamana", "data_type": data_type, "svs-vamana_config": svs_config}


def fixed_points(top, knob, parallel):
    return [
        {"parallel": parallel, "top": top, "search_params": {knob: breadth}}
        for breadth in breadths
    ]


def calibrated_points(top, knob, parallel):
    return [
        {
            "parallel": parallel,
            "top": top,
            "calibration_param": knob,
            "calibration_precision": target,
            "search_params": {},
        }
        for target in calibration_targets
    ]


def k10_points():
    return [p for c in clients for p in fixed_points(10, "ef", c)]


def comparison_points(knob):
    return [
        p
        for top in comparison_topKs
        for c in clients
        for p in fixed_points(top, knob, c) + calibrated_points(top, knob, c)
    ]


def make_case(name, data_type, collection_params, search_params):
    return {
        "name": name,
        "engine": "redis",
        "collection_params": collection_params,
        "upload_params": {"parallel": 8, "batch_size": 64, "data_type": data_type},
        "search_params": search_params,
    }


def write_configs(fname, configs):
    lines = []
    for config in configs:
        fields = [
            f'    "{key}": {json.dumps(value)},'
            for key, value in config.items()
            if key != "search_params"
        ]
        points = ",\n".join(f"      {json.dumps(p)}" for p in config["search_params"])
        lines.append(
            "  {\n" + "\n".join(fields) + '\n    "search_params": [\n' + points + "\n    ]\n  }"
        )
    with open(os.path.join(out_dir, fname), "w") as json_fd:
        json_fd.write("[\n" + ",\n".join(lines) + "\n]\n")
    print(f"Created {len(configs)} configs for {fname}.")


hnsw_variants = [("plain", None)] + list(sq8_training_thresholds.items())

k10_configs = []
comparison_configs = []
for data_type in data_types:
    dt = data_type.lower()
    for variant, threshold in hnsw_variants:
        collection = hnsw_collection(data_type, threshold)
        k10_configs.append(
            make_case(f"redis-hnsw-{dt}-{variant}-k10", data_type, collection, k10_points())
        )
        comparison_configs.append(
            make_case(
                f"redis-compare-{dt}-{variant}", data_type, collection, comparison_points("ef")
            )
        )
    comparison_configs.append(
        make_case(
            f"redis-compare-{dt}-svs-{svs_compression.lower()}",
            data_type,
            svs_collection(data_type),
            comparison_points("SEARCH_WINDOW_SIZE"),
        )
    )

write_configs("redis-hnsw-sq8-k10.json", k10_configs)
write_configs("redis-svs-sq8-comparison.json", comparison_configs)
