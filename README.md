# OpenCLIP Image Retrieval Benchmarking

This project benchmarks zero-shot image retrieval using OpenCLIP ViT-B-32 pretrained on LAION-2B (2 billion image-text pairs) on a custom dataset of 451,958 images across 100 classes. No fine-tuning is performed — embeddings are extracted directly from the pretrained model using 512-dimensional vectors and evaluated using FAISS-accelerated cosine similarity, demonstrating that CLIP achieves competitive performance against a fine-tuned ResNet50 with zero task-specific training.

## Results
| Metric | Score |
|--------|-------|
| Precision@1 | 89.38% |
| Precision@5 | 88.48% |
| kNN Accuracy @21 | 91.16% |
