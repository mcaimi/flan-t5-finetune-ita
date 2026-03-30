#!/usr/bin/env python
#
# Download Google FLAN-T5 checkpoint from HuggingFace

try:
    import dotenv
    import huggingface_hub as hf
    from libs.utility import set_proxy, downloadFromHuggingFace
    from libs.parameters import Properties
except ImportError as e:
    print(f"Error: {e}")
    raise e

# load config environment via dotenv
config_env = dotenv.dotenv_values("localenv")
params = Properties(config_env.get("PARAMETER_FILE"))

PROXIES: dict = {
    "http": config_env.get("HTTP_PROXY", ""),
    "https": config_env.get("HTTPS_PROXY", ""),
}

# setup proxy
set_proxy(proxies=PROXIES)

print(f"Using proxy: HTTP={os.environ.get('HTTP_PROXY')} or HTTPS={os.environ.get('HTTPS_PROXY')}")

model_path = downloadFromHuggingFace(
    repo_id="google/flan-t5-small",
    local_dir=params.config_parameters.huggingface.local_dir,
    cache_dir=params.config_parameters.huggingface.cache_dir,
    apitoken=params.config_parameters.huggingface.apitoken,
)
