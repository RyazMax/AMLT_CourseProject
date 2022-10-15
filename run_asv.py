import torch
import torchaudio
from torch.utils import data
from typing import Tuple
import os
from pyeer.eer_info import get_eer_stats

from tqdm import tqdm
from transformers import Wav2Vec2Processor, HubertModel
from torchmetrics import WordErrorRate
from asv_model import SpeakerExtractor


VOXCELEB_PATH = '/mnt/cs/datasets/Voxceleb-o/test'
VOXCELEB_CLASSES = 1241

def load_item(line: str, path: str, folder_audio: str, ext_audio: str = '.wav') -> Tuple[torch.Tensor, int, str, str]:

    enroll_id, test_id, label = line[0].strip().split(" ")

    enroll_audio = os.path.join(path, folder_audio, enroll_id + ext_audio)
    test_audio = os.path.join(path, folder_audio, test_id + ext_audio)

    # Load audio
    enroll_waveform, _ = torchaudio.load(enroll_audio)
    test_waveform, _ = torchaudio.load(test_audio)

    return (enroll_waveform, test_waveform, label)


class ASVDataSet(data.Dataset):
    def __init__(self, path):
        self.path = path
        with open(f'{self.path}/meta.txt', 'r') as text:
            self.meta = text.readlines()

    def __getitem__(self, index):
        line = self.meta[index]
        return load_item(line, self.path, '.')

    def __len__(self):
        return len(self.meta)        


if __name__ == "__main__":

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = SpeakerExtractor(hubert_model, VOXCELEB_CLASSES)

    ckpt_path = "checkpoint_100_legacy.pt" # Checkpoint from https://github.com/auspicious3000/contentvec
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    cvec_model = SpeakerExtractor(models[0], VOXCELEB_CLASSES)

    models = [hubert_model, cvec_model]
    ami_ds = ASVDataSet(VOXCELEB_PATH)
    dl = data.DataLoader(ami_ds, batch_size=1, num_workers=8)

    for model in models:
        preds, targs = [], []
        model.eval()
        for e_feats, test_feats, label in tqdm(dl, desc='evaluating'):
            e_inputs = processor(e_feats, return_tensors="pt").input_values
            t_inputs = processor(e_feats, return_tensors="pt").input_values
            with torch.no_grad():
                e_emb = model(e_feats)
                t_emb = model(t_inputs)
            score = e_emb.dot(t_emb)

            preds += [score]
            targs += [label]

        # calc EER
        eer = get_eer_stats(
            [score for score, label in zip(preds, targs) if label == 1],
            [score for score, label in zip(preds, targs) if label == 0],
        )
        print('EER: ', eer)
