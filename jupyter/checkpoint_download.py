#!/usr/bin/env python
#
# Download Google FLAN-T5 checkpoint from HuggingFace

try:
    import dotenv
    import huggingface_hub as hf
    from libs.utility import downloadFromHuggingFace
    from libs.parameters import Properties
except ImportError as e:
    print(f"Error: {e}")
    raise e

# load config environment via dotenv
config_env = dotenv.dotenv_values("localenv")
params = Properties(config_env.get("PARAMETER_FILE"))

model_path = downloadFromHuggingFace(
    repo_id="google/flan-t5-small",
    local_dir=params.config_parameters.huggingface.local_dir,
    cache_dir=params.config_parameters.huggingface.cache_dir,
    apitoken=params.config_parameters.huggingface.apitoken,
)
