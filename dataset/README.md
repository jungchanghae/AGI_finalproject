## We use Image-Net Hard

1. Using dataset in huggingface

    ```python
    from datasets import load_dataset, Image
    dataset = load_dataset("taesiri/imagenet-hard", split='validation')
    ```

2. Download from HuggingFace to local

    change directory and ``` python huggingface_data_download.py ```