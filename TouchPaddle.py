# Download required PaddleOCR models and fonts.
# Also imported by MagiaTimeline.py to apply the langchain compat shim before
# anything imports paddleocr.

import os
os.environ["PADDLE_PDX_CACHE_HOME"] = "./PaddleOCRModels"

# paddlex 3.1.3 eagerly imports its full pipeline tree including
# retriever/base.py, which imports `langchain.docstore` and `langchain.text_splitter`.
# These paths were removed in langchain 1.x (moved to langchain_core and
# langchain_text_splitters). Inject compatibility shims now before any paddle
# import in this process so the `if is_dep_available("langchain")` guards in
# paddlex proceed without error. setdefault leaves langchain 0.3.x untouched.
import sys
import types

_docstore_mod = types.ModuleType("langchain.docstore")
_document_mod = types.ModuleType("langchain.docstore.document")
try:
    from langchain_core.documents import Document as _Document
    _document_mod.Document = _Document
except ImportError:
    pass
_docstore_mod.document = _document_mod
sys.modules.setdefault("langchain.docstore", _docstore_mod)
sys.modules.setdefault("langchain.docstore.document", _document_mod)

_ts_mod = types.ModuleType("langchain.text_splitter")
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _RCS
    _ts_mod.RecursiveCharacterTextSplitter = _RCS
except ImportError:
    pass
sys.modules.setdefault("langchain.text_splitter", _ts_mod)

if __name__ == "__main__":
    import paddlex.inference.utils.official_models
    import paddlex.utils.fonts # Trigger download of default fonts
    import logging

    logging.basicConfig(level=logging.INFO)

    paddlex.inference.utils.official_models.official_models["PP-OCRv4_mobile_det"]
    paddlex.inference.utils.official_models.official_models["PP-OCRv5_mobile_rec"]
