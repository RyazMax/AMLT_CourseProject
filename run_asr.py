import torch
import torchaudio
from torch.utils import data
from typing import Tuple
import os

from tqdm import tqdm
from transformers import Wav2Vec2Processor, HubertForCTC
from torchmetrics import WordErrorRate


AMI_PATH = '/mnt/cs/datasets/AMI/test'

def load_item(line: str, path: str, folder_audio: str, ext_audio: str = '.wav') -> Tuple[torch.Tensor, int, str, str]:

    utterance_id, transcript = line[0].strip().split(" ", 2)[1:]

    # Remove space, double quote, and single parenthesis from transcript
    transcript = transcript[1:-3]

    file_audio = os.path.join(path, folder_audio, utterance_id + ext_audio)

    # Load audio
    waveform, _ = torchaudio.load(file_audio)

    return (waveform, transcript)


class ASRDataSet(data.Dataset):
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
    hubert_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

    ckpt_path = "checkpoint_100_legacy.pt" # Checkpoint from https://github.com/auspicious3000/contentvec
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    cvec_model = models[0]

    models = [hubert_model, cvec_model]
    ami_ds = ASRDataSet(AMI_PATH)
    dl = data.DataLoader(ami_ds, batch_size=1, num_workers=8)

    for model in models:
        preds, targs = [], []
        for feats, labels in tqdm(dl, desc='evaluating'):
            inputs = processor(feats, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=1)

            res = processor.batch_decode(predicted_ids)

            preds += res
            targs += labels

        # calc WER
        print('WER: ', WordErrorRate()(preds, targs))
