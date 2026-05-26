# OpenCLIP ViT-H/14 Loader

Loads OpenCLIP ViT-H/14 pretrained on LAION-2B for zero-shot embedding extraction. No training required.

## Model

OpenCLIP ViT-H/14 with 632M parameters pretrained on LAION-2B (2 billion image-text pairs) using contrastive language-image pretraining. The Vision Transformer splits images into 14×14 patches and learns relationships between all patches simultaneously through self-attention. Embeddings of 1024 dimensions are extracted directly from the pretrained image encoder without any fine-tuning.

## Usage

The script outputs `model`, `train_loader`, `val_loader` and `test_loader` ready to import into `extractor.py`. OpenCLIP's own preprocessing pipeline is used — no custom transforms needed.

```python
from clip_loader import model, train_loader, val_loader, test_loader
```

Batch size is set to 8 due to the large model size. To change the cache location for model weights:

```python
os.environ['HF_HOME'] = '/your/path/here'
```

## Key Details

| Property | Value |
|----------|-------|
| Architecture | ViT-H/14 |
| Pretrained on | LAION-2B |
| Parameters | 632M |
| Embedding dim | 1024 |
| Batch size | 8 |
| Fine-tuning | None (zero-shot) |
