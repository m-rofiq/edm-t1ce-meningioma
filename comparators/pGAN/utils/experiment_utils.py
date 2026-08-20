import os
import json


def prepare_experiment(CONFIG):

    exp_root = CONFIG["experiment_root"]

    exp_name = f'{CONFIG["experiment_id"]}_{CONFIG["experiment_name"]}_fold{CONFIG["fold"]}'

    exp_dir = os.path.join(exp_root, exp_name)

    os.makedirs(exp_dir, exist_ok=True)

    # save config snapshot
    config_path = os.path.join(exp_dir, "config_snapshot.json")

    if not os.path.exists(config_path):

        with open(config_path, "w") as f:
            json.dump(CONFIG, f, indent=4)

    return exp_dir