from classification.shard_dataset import ShardDataset  
import yaml
from classification.write_shards import MsgPackWriter
import torch  
import torchvision.transforms as T

with open('dataset_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print("Config loaded")
model_params = config["writer_params"]
rgb_params = config["transform_params"]          

dataset = ShardDataset(**model_params)

print("Data loader created")


rgb_dataset = ShardDataset(source_dir="/rds/general/user/ao921/ephemeral/Transformer_Based_Geo-localization/resources/shards/yfcc25600",
                          meta_path="/rds/general/user/ao921/ephemeral/Transformer_Based_Geo-localization/resources/yfcc25600_places365.csv")

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config["loader_params"]["batch_size"],
    num_workers=config["loader_params"]["num_workers"],
    pin_memory=True,
    drop_last = True,
    )
rgb_data_loader = torch.utils.data.DataLoader(
        rgb_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last = True,
    )
rgb_it = iter(rgb_data_loader)
it = iter(data_loader)

items  = dict()

for entry in it:
    items[entry[1][0]] = entry

missing = []

for entry in rgb_it:
    if entry[1][0] in items:
        items[entry[1][0]].append(entry[0])
    else:
        missing.append(entry[1][0])
        
        
dataset = []
for key in items.keys():
    item = items[key]
    if len(item) == 5:
        transform = T.Resize(size = (256,340))
        resize = T.Resize(size = (256,340), interpolation=Image.NEAREST)
        seg = resize(item[0]).numpy()
        line  = dict()
        line["segmentation"] = seg[0]
        line["image_id"] = item[1][0]
        line["lat"] =  item[2][0].numpy()    
        line["lon"] =  item[3][0].numpy()
        line["image"] =  transform(item[4][0]).numpy()
        dataset.append(line)
        
with MsgPackWriter(config["combined_dataset"]) as writer:
    for batch in dataset:
        writer.write(batch)
