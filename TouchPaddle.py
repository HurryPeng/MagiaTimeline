# Download required PaddleOCR models

import paddlex
import paddlex.inference.utils.official_models
import paddlex.utils.download
import pathlib
import logging

# A reimplementation of paddlex.inference.utils.official_models.OfficialModelsDict that allows
# downloading and caching models in a specified directory.
class OfficialModelsDict(dict):
    def __init__(self, d, cacheDir: str):
        super().__init__(d)
        self.cacheDir = cacheDir

    def __getitem__(self, key):
        url = super().__getitem__(key)
        save_dir = pathlib.Path(self.cacheDir) / "official_models"
        logging.info(
            f"Using official model ({key}), the model files will be automatically downloaded and saved in {save_dir}."
        )
        paddlex.utils.download.download_and_extract(url, save_dir, f"{key}", overwrite=False)
        return save_dir / f"{key}"

official_models = OfficialModelsDict(paddlex.inference.utils.official_models.OFFICIAL_MODELS, cacheDir="./PaddleOCRModels")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    official_models["PP-OCRv4_mobile_det"]
    official_models["PP-OCRv5_mobile_rec"]
