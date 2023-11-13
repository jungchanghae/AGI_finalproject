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
        labels = df['english_label'][i]
        for label in labels:
            os.makedirs('/workspace/project/DATA/imagenet_hard/images/{}'.format(label),exist_ok=True)
            image = Image.open(io.BytesIO(df['image'][i]['bytes']))
            image.convert("RGB").save(dataset_dir+label+'/{}_{}_{}.png'.format(p,label,i))
            
### Class name txt
import requests

url = "https://gist.githubusercontent.com/taesiri/5b5edb5452f2f20d82d5ed1bb58ab574/raw/0376003d3999799f99208fb338cfd06ee44372b5/imagenet-labels.json"

response = requests.get(url)

if response.status_code == 200:
    imagenet_classes = response.json()
    
file_name = '/workspace/project/DATA/imagenet_hard/classnames.txt'

with open(file_name, 'w+') as file:
    file.write('\n'.join(imagenet_classes)) 