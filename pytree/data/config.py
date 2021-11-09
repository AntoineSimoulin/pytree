import copy
import json
from typing import Any, Dict, Tuple, Union


class Config:

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.
        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
       
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type 

        self.dict_torch_dtype_to_str(output)

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary has a `torch_dtype` key and if it's not None, converts torch.dtype to a
        string of just the type. For example, :obj:`torch.float32` get converted into `"float32"` string, which can
        then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
