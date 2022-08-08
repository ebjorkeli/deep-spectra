import json

def get_config(config_file):
    with open(config_file) as f:
        data = f.read()
    config = json.loads(data)
    return config