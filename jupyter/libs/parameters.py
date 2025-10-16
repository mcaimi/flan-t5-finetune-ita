#!/usr/bin/env python

try:
    import os
    from yaml import safe_load, YAMLError
except ImportError as e:
    raise(e)

# configuration params wrapper
class Parameters(object):
    def __init__(self, data: dict):
        if type(data) != dict:
            raise TypeError(f"Parameters: expected 'dict', got {type(data)}.")
        else:
            self.data = data

        for k in self.data.keys():
            if type(self.data.get(k)) != dict:
                self.__setattr__(k, self.data.get(k))
            else:
                self.__setattr__(k, Parameters(self.data.get(k)))

# settings class, wraps many configuration aspects of the LLM application
class Properties(object):
    def __init__(self, config_file: str) -> None:
        self.config_file_name = config_file
        try:
            # init session state tracker
            self.config_parameters: Parameters = None
            try:
                self.load_config_parms()
            except Exception as e:
                raise e

            # perform sanity check
            self.bootup_check()
        except Exception as e:
            raise e

    def load_config_parms(self) -> None:
        try:
            with open(self.config_file_name, "r") as f:
                config_parms = safe_load(f)

            self.config_parameters = Parameters(config_parms)
        except YAMLError as e:
            raise e
        except Exception as e:
            raise e

    # sanity check at bootup
    def bootup_check(self) -> None:
        os.makedirs(self.config_parameters.huggingface.local_dir, exist_ok=True)
        os.makedirs(self.config_parameters.huggingface.cache_dir, exist_ok=True)

    # session variables
    def get_properties_object(self) -> dict:
        return self.config_parameters