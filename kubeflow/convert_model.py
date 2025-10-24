# Import objects from KubeFlow DSL Library
from kfp.dsl import (
    component,
    Input,
    Output,
    Model,
)


@component(base_image="python:3.11",
          packages_to_install=["optimum", "transformers", "optimum[onnxruntime]"])
def convert_model(
    checkpoint_dir: str,
    finetuned_model: Input[Model],
    onnx_model: Output[Model],
):
    from zipfile import ZipFile
    from pathlib import Path
    import os

    # import libraries
    try:
        import torch
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"Exception during library import {e}")

    # local dir
    WORKDIR: str = f"{checkpoint_dir}/workdir"
    os.makedirs(WORKDIR, exist_ok=True)
    # create output dir
    ONNX_DIR: str = f"{checkpoint_dir}/onnx"
    os.makedirs(ONNX_DIR, exist_ok=True)

    # decompress finetuned model
    with ZipFile(finetuned_model.path, 'r') as ftuned:
        ftuned.extractall(WORKDIR)

    # load model from local path via Optimum ONNX Optimizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(WORKDIR)
        model = ORTModelForSeq2SeqLM.from_pretrained(
            WORKDIR,
            export=True
        )
    except Exception as e:
        print(f"Exception during model export: {e}")

    # save onnx to disk
    try:
        model.save_pretrained(ONNX_DIR)
        tokenizer.save_pretrained(ONNX_DIR)
    except Exception as e:
        print(e)

    # clean up
    del model
    del tokenizer

    # save model to s3
    onnx_model._set_path(onnx_model.path + "-onnx.zip")
    onnx_path = Path(ONNX_DIR)

    # zip & store
    import zipfile
    with zipfile.ZipFile(onnx_model.path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in onnx_path.rglob("*"):
            zip_file.write(entry, entry.relative_to(onnx_path))