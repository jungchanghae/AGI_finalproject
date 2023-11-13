import os
import io
import PIL.Image as Image

from array import array
import pandas as pd 
import huggingface_hub

# raw_data directory
raw_data_path = '/workspace/data/'

huggingface_hub.snapshot_download('taesiri/imagenet-hard',repo_type='dataset', local_dir=raw_data_path)

# image data directory
dataset_dir = '/workspace/project/DATA/imagenet_hard/images/'

for p in os.listdir(raw_data_path+'/data/'):
    df = pd.read_parquet(raw_data_path+'/data/'+p, engine='pyarrow') 
    for i in range(len(df)):
        label = df['english_label'][i][0]
        os.makedirs('/workspace/project/DATA/imagenet_hard/images/{}'.format(label),exist_ok=True)
        image = Image.open(io.BytesIO(df['image'][0]['bytes']))
        image.save(dataset_dir+label+'/{}_{}.jpg'.format(p,i))