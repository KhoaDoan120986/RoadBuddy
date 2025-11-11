from torch.utils.data import Dataset
import torch
import h5py
import json
from tqdm import tqdm


class VideoTextDataset(Dataset):
    def __init__(self, video_feature_path, text_feature_path, text_metadata_path, max_choices=4):
        self.video_feature_path = video_feature_path
        self.text_feature_path = text_feature_path
        self.text_metadata_path = text_metadata_path
        self.max_choices = max_choices

        with open(self.text_metadata_path, "r", encoding="utf-8") as f_json:
            self.text_metadata = json.load(f_json)

        self.question_ids = []
        self.video_ids = set()
        for key, value in self.text_metadata.items():
            self.question_ids.append(key)
            self.video_ids.add(value['video_id'])

        self.video_features = {}
        self.max_video_tokens = -1
        with h5py.File(self.video_feature_path, "r") as f_video:
            for key in tqdm(f_video.keys(), desc="Loading video features..."):
                feat = f_video[key][()]
                self.max_video_tokens = max(self.max_video_tokens, feat.shape[0])
                if key in self.video_ids:
                    self.video_features[key] = feat

        self.text_features = {}
        self.max_text_tokens = -1
        with h5py.File(self.text_feature_path, "r") as f_text:
            for key in tqdm(f_text.keys(), desc="Loading text features..."):
                feat = f_text[key][()] 
                self.max_text_tokens = max(self.max_text_tokens, feat.shape[1])
                self.text_features[key] = feat


    def __len__(self):
        return len(self.question_ids)

    def __getitem__(self, idx):
        qid = self.question_ids[idx]
        meta = self.text_metadata[qid]
        vid_id = meta.get("video_id", None)
        answer_idx = meta.get("answer", None)

        video_feat = torch.tensor(self.video_features[vid_id], dtype=torch.float32)
        video_len, embed_dim_v = video_feat.shape

        video_feat_padded = torch.zeros((self.max_video_tokens + 1, embed_dim_v), dtype=torch.float32)
        video_mask = torch.zeros(self.max_video_tokens + 1, dtype=torch.float32)

        video_feat_padded[1:video_len+1] = video_feat[:self.max_video_tokens]
        video_mask[0] = 1
        video_mask[1:video_len+1] = 1 

        text_feat = torch.tensor(self.text_features[qid], dtype=torch.float32)
        num_choices, num_tokens, embed_dim_t = text_feat.shape

        padded_text = torch.zeros((self.max_choices, self.max_text_tokens + 1, embed_dim_t), dtype=torch.float32)
        text_attention_mask = torch.zeros((self.max_choices, self.max_text_tokens + 1), dtype=torch.float32)

        valid_choices = min(num_choices, self.max_choices)
        for i in range(valid_choices):
            t = text_feat[i]
            tok_len = min(t.shape[0], self.max_text_tokens)
            padded_text[i, 1:tok_len+1] = t[:tok_len]
            text_attention_mask[i, 0] = 1 
            text_attention_mask[i, 1:tok_len+1] = 1

        answer_vec = torch.zeros(self.max_choices, dtype=torch.float32)
        if answer_idx is not None and 0 <= answer_idx < self.max_choices:
            answer_vec[answer_idx] = 1.0

        return vid_id, video_feat_padded, video_mask, padded_text, text_attention_mask, answer_vec