from code.data_loader import RWFFightDataset
from torch.utils.data import DataLoader

csv_path = "data/metadata/rwf2000_metadata.csv"

train_ds = RWFFightDataset(csv_path, split="train", clip_len=8, img_size=224)
print("Train videos:", len(train_ds))

clip, label = train_ds[0]
print("Clip shape:", clip.shape)   # expected: [T, C, H, W] = [8, 3, 224, 224]
print("Label:", label)
