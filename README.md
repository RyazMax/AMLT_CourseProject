# AMLT_CourseProject
Advanced machine learning technologies course project in ITMO. Evaluating ContentVec https://github.com/auspicious3000/contentvec

To run ASR evaluation on AMI corpus run:

```sh
python3 run_asr.py
```

script will evaluate WER quality of HuBERT and ContentVec models

To run ASV evaluation on VoxCeleb corpus run:

```sh
python3 run_asv.py
```

This will calculate EER of models


The fine-tuned architecture for ASV task in the `asv_model.py` file.

>> Note that the evaluation dataset are not in repo. So place data where it is convient for you and change constants in the begining of scripts.
