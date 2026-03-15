

import os
import shutil
import random
import string
import pandas as pd
from enum import Enum
from typing import Literal


class Lang(str, Enum):
    ENGLISH = "English"
    URDU = "Urdu"
    HINDI = "Hindi"
    GERMAN = "German"

class Version(str, Enum):
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"

random.seed(42)
KEY_LEN = 10

SupportedLangs = Literal["English", "Urdu", "Hindi", "German"]

split = "val"
version: Version = Version.V1
test_lang: Lang = Lang.ENGLISH

def build_dst_audio(row):
    src = row["audio_path"]

    parts = src.split("/")

    version = parts[-6]
    modality = parts[-5]
    # identity = parts[-4]
    lang = parts[-3]

    idx = f"{row.name+1:05d}.wav"

    return os.path.join(
        f"./data/{split}/",
        version,
        modality,
        # identity,
        lang,
        idx
    )


def build_dst_face(row):
    src = row["face_path"]

    parts = src.split("/")

    version = parts[-6]
    modality = parts[-5]
    # identity = parts[-4]
    lang = parts[-3]

    idx = f"{row.name+1:05d}.jpg"

    return os.path.join(
        f"./data/{split}/",
        version,
        modality,
        # identity,
        lang,
        idx
    )

def build_dst_feats(row, src_col):
    src = row[src_col]
    parts = src.replace("\\","/").split("/")
    version = parts[-6]
    modality = parts[-5]
    lang = parts[-3]
    
    idx = f"{row.name+1:05d}.npy"
    
    return os.path.join(
        f"./features/{split}",
        version,
        modality,
        lang,
        idx
        )

path = "./feature_tracker/"
data_root = "../"
dst_dir = f"./{version.value}"
dst_dir_feats = "./feats/"
test_csv_tracker_path = os.path.join(path, 
                    f"{version.value}_{split}_{test_lang.value}.csv")
test_csv = pd.read_csv(test_csv_tracker_path)

sub_dict = test_csv[["audio_path", "face_path", "identity", 
                     "ecappa_feats_path", "facenet_feats_path"]].copy()
sub_dict = sub_dict.sample(frac=1).reset_index(drop=True)
sub_dict["label"] = sub_dict["identity"].apply(
    lambda i: int(i.split("id")[1])-1)
sub_dict["audio_path"] = sub_dict["audio_path"].apply(
    lambda p: os.path.join(data_root, p.lstrip("./")))
sub_dict["face_path"] = sub_dict["face_path"].apply(
    lambda p: os.path.join(data_root, p.lstrip("./")))

sub_dict["dst_audio_path"] = sub_dict.apply(build_dst_audio, axis=1)
sub_dict["dst_face_path"] = sub_dict.apply(build_dst_face, axis=1)

sub_dict["ecappa_feats_path"] = sub_dict["ecappa_feats_path"].apply(
    lambda p: os.path.join("../",  p.lstrip("./")))
sub_dict["facenet_feats_path"]= sub_dict["facenet_feats_path"].apply(
    lambda p: os.path.join("../",  p.lstrip("./")))

sub_dict["dst_audio_feats"] = sub_dict.apply(
    lambda r: build_dst_feats(r, "ecappa_feats_path"),
    axis=1
)
sub_dict["dst_face_feats"] = sub_dict.apply(
    lambda r: build_dst_feats(r, "facenet_feats_path"),
    axis=1
)

chars = string.ascii_letters + string.digits

keys = set()
while len(keys) < len(sub_dict):
    keys.add("".join(random.choices(chars, k=KEY_LEN)))
key_list = list(keys)

sub_dict.insert(0, "key", key_list)

to_remove = set()
for row in sub_dict.itertuples(index=False):
    src_face = row.face_path
    src_audio = row.audio_path
    dst_face = row.dst_face_path
    dst_audio = row.dst_audio_path
    
    os.makedirs(os.path.dirname(dst_face), exist_ok=True)
    os.makedirs(os.path.dirname(dst_audio), exist_ok=True)
    
    src_audio_feat = row.ecappa_feats_path
    src_face_feat = row.facenet_feats_path
    dst_audio_feat = row.dst_audio_feats
    dst_face_feat = row.dst_face_feats
    
    os.makedirs(os.path.dirname(dst_audio_feat), exist_ok=True)
    os.makedirs(os.path.dirname(dst_face_feat), exist_ok=True)
    
    shutil.move(src_face, dst_face)
    shutil.move(src_audio, dst_audio)  
    shutil.move(src_audio_feat, dst_audio_feat)
    shutil.move(src_face_feat, dst_face_feat)
    
    to_remove.add(os.path.dirname(src_face))
    to_remove.add(os.path.dirname(src_audio))
    to_remove.add(os.path.dirname(src_audio_feat))
    to_remove.add(os.path.dirname(src_face_feat))

for rem in to_remove:
    shutil.rmtree(rem)

sub_dict.to_csv(f"{version.value}_{test_lang.value}_{split}_key_dict.csv",index=None)

sub_dict["voices"] = sub_dict["dst_audio_path"].apply(
    lambda p: p.replace("\\", "/").replace("./data/", ""))

sub_dict["faces"] = sub_dict["dst_face_path"].apply(
    lambda p: p.replace("\\", "/").replace("./data/", ""))

sub_dict[["key", "voices", "faces"]].to_csv(f"{version.value}_{split}_{test_lang.value}.csv",index=None)
