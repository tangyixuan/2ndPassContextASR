# 2ndPassContextASR

This repository contains the code for the paper [Contextualized Speech Recognition: Rethinking Second-Pass Rescoring with Generative Large Language Models](https://www.ijcai.org/proceedings/2024/0716.pdf) published in IJCAI 2024.

## Overview

Our approach leverages contextual information to enhance speech recognition, particularly when dealing with diverse accents, as demonstrated on the [SQuAD-SRC](https://github.com/tangyixuan/SQuAD-SRC) dataset.

<img width="679" alt="image" src="https://github.com/user-attachments/assets/02614c6b-4497-4da9-a7e5-6bc934c1fea6">

Key features include:
- Utilizing pre-trained speech models to generate diverse transcription candidates.
- Exploiting contextual information for on-the-fly in-domain adaptation.
- Employing large language models to refine transcriptions with rich linguistic knowledge.

## Usage

### Zero-Shot Prompting
To run zero-shot prompting:
```bash
python run_regenerate.py
```
<img width="524" alt="image" src="https://github.com/user-attachments/assets/0199083e-b6ad-406c-a22b-f52e21758d18">

This method achieves a 13.6% performance improvement without tuning the pre-trained speech and language models.

### LoRA Tuning
To perform LoRA tuning:
1. Tune the model
2. Run the re-generation using finetuned model

```bash
python peft.py
python run_regenerate.py
```

<img width="495" alt="image" src="https://github.com/user-attachments/assets/e7d00091-45ab-4c5d-a36a-ee298fec14af">

Results show consistent performance gains with increasing training examples:
- Tuning with just 100 examples results in a 19.8% improvement with Whisper Tiny and a 12% improvement with Whisper Medium.
- Whisper Tiny tuned with 500 examples can outperform Whisper Medium, despite having about 20x fewer parameters.
