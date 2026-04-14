# Download required PaddleOCR models and fonts

import os
os.environ["PADDLE_PDX_CACHE_HOME"] = "./PaddleOCRModels"

if __name__ == "__main__":
    import paddlex.inference.utils.official_models
    import paddlex.utils.fonts # Trigger download of default fonts
    import logging

    logging.basicConfig(level=logging.INFO)

    paddlex.inference.utils.official_models.official_models["PP-OCRv4_mobile_det"]
    paddlex.inference.utils.official_models.official_models["PP-OCRv5_mobile_rec"]
