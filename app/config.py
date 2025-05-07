import os

GLOBAL_CONFIG = {
    "MODEL_PATH": "../model/bert_regressor.onnx",
    "BERT_MODEL": "answerdotai/ModernBERT-base",
    "MAX_LEN": 2048,
    "DEVICE": "cpu"
}

ENV_CONFIG = {
    "development": {
        "DEBUG": True
    },

    "staging": {
        "DEBUG": True
    },

    "production": {
        "DEBUG": False,
        "ROUND_DIGIT": 3
    }
}

def get_config() -> dict:
    """
    Get config based on running environment
    :return: dict of config
    """

    # Determine running environment
    ENV = os.environ['PYTHON_ENV'] if 'PYTHON_ENV' in os.environ else 'development'
    ENV = ENV or 'development'

    # raise error if environment is not expected
    if ENV not in ENV_CONFIG:
        raise EnvironmentError(f'Config for envirnoment {ENV} not found')

    config = GLOBAL_CONFIG.copy()
    config.update(ENV_CONFIG[ENV])

    config['ENV'] = ENV

    return config

CONFIG = get_config()

if __name__ == '__main__':
    # for debugging
    import json
    print(json.dumps(CONFIG, indent=4))