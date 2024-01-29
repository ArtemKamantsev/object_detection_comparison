import dataclasses
import json
import os
from typing import Any, Union, List

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return super().encode(bool(obj))
        elif isinstance(obj, np.float32) or isinstance(obj, np.float16):
            return super().encode(float(obj))
        elif dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)

        return super().default(obj)


def map_from_json(value: Union[str, None]) -> Any:
    if pd.isna(value):
        return None

    return json.loads(value)


def map_to_json(value: object) -> str:
    if pd.isna(value) is True:
        return np.nan

    return json.dumps(value, cls=CustomJSONEncoder)


def map_data_frame_from_json(data_frame: pd.DataFrame, column_list: List[str]) -> pd.DataFrame:
    for c in column_list:
        data_frame[c] = data_frame[c].map(map_from_json)

    return data_frame


def map_data_frame_to_json(data_frame: pd.DataFrame, column_list: List[str]) -> pd.DataFrame:
    for c in column_list:
        data_frame[c] = data_frame[c].map(map_to_json)

    return data_frame


def get_file_name_list(folder_path: str) -> List[str]:
    file_name_list = os.listdir(folder_path)
    file_name_list = [name for name in file_name_list if name[0] != '.']  # filter files with names like '.DS_Store'

    return file_name_list


def show(img: np.ndarray, dpi: int = 300, convert_rgb: bool = True) -> None:
    if convert_rgb:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(dpi=dpi)
    plt.imshow(img)
    plt.show()
