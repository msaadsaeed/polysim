import os
import torch
import numpy as np
import pandas as pd

from config import ExperimentConfig
from models.fop import FOP


def load_npy(csv_file, feats_dir, device):
    csv_file["ecappa_feats_path"] = csv_file["voices"].apply(
        lambda p: os.path.join(feats_dir, p).replace(".wav", ".npy"))
    csv_file["facenet_feats_path"] = csv_file["faces"].apply(
        lambda p: os.path.join(feats_dir, p).replace(".jpg", ".npy"))
    
    voice_feats = [np.load(i) for i in csv_file["ecappa_feats_path"]]
    face_feats = [np.load(i) for i in csv_file["facenet_feats_path"]]
    
    voice_feats = np.asarray(voice_feats)
    face_feats = np.asarray(face_feats)
    
    voice_feats = torch.from_numpy(voice_feats).to(device)
    face_feats = torch.from_numpy(face_feats).to(device)
    
    return voice_feats, face_feats

def main():
    # --------------------------------------------------
    # Config
    # --------------------------------------------------
    config = ExperimentConfig()
    config.debug = False
    device = torch.device(config.device)

    torch.manual_seed(config.seed)
    
    SPLIT = "val"
    FEATS_DIR = "./features"
    UNSEEN_TEST_LANG = "English" if config.seen_lang== "Urdu" else "Urdu"
    
    # --------------------------------------------------
    # Load test dataset (in-memory)
    # --------------------------------------------------
    test_csv = pd.read_csv(f"./csv_files/comp/{config.version}_{SPLIT}_{config.seen_lang}.csv")
    unseen_test_csv = pd.read_csv(f"./csv_files/comp/{config.version}_{SPLIT}_{UNSEEN_TEST_LANG}.csv")
    
    seen_voice_feats, seen_face_feats = load_npy(test_csv, FEATS_DIR, device)
    unseen_voice_feats, unseen_face_feats = load_npy(unseen_test_csv, FEATS_DIR, device)
    
    
    # # --------------------------------------------------
    # # Load model
    # # --------------------------------------------------
    face_dim = seen_face_feats.shape[1]
    audio_dim = seen_voice_feats.shape[1]

    model = FOP(
        config=config,
        face_dim=face_dim,
        voice_dim=audio_dim,
    ).to(device)

    checkpoint_path = f"./checkpoints/{config.version}_{config.seen_lang}_alpha0.0_best.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # # --------------------------------------------------
    # # P3
    # # --------------------------------------------------
    _, logits, _, _ = model(seen_face_feats, seen_voice_feats)
    p3 = logits.argmax(dim=1).detach().cpu().numpy()
    
    # # --------------------------------------------------
    # # P4
    # # --------------------------------------------------
    _, logits, _, _ = model(seen_face_feats*0.0, seen_voice_feats)
    p4 = logits.argmax(dim=1).detach().cpu().numpy()
    
    # # --------------------------------------------------
    # # P5
    # # --------------------------------------------------
    _, logits, _, _ = model(unseen_face_feats, unseen_voice_feats)
    p5 = logits.argmax(dim=1).detach().cpu().numpy()
    
    # # --------------------------------------------------
    # # P6
    # # --------------------------------------------------
    _, logits, _, _ = model(unseen_face_feats*0.0, unseen_voice_feats)
    p6 = logits.argmax(dim=1).detach().cpu().numpy()
    
    submission = pd.DataFrame()
    submission["key"] = test_csv["key"]
    submission["p3"] = p3
    submission["p4"] = p4
    submission.to_csv(f"csv_files/submission/submission_{config.version}_{SPLIT}_{config.seen_lang}_{config.seen_lang}.csv", index=None)
    submission = pd.DataFrame()
    submission["key"] = unseen_test_csv["key"]
    submission["p5"] = p5
    submission["p6"] = p6   
    submission.to_csv(f"csv_files/submission/submission_{config.version}_{SPLIT}_{config.seen_lang}_{UNSEEN_TEST_LANG}.csv", index=None)
    
if __name__ == "__main__":
    main()
