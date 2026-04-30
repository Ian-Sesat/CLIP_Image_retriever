import open_clip
import torch
import os
import warnings
import numpy as np
from tqdm import tqdm
import faiss


from PIL import ImageFile
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load OpenCLIP 
model, _, preprocess_val = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = model.to(device)
model.eval()
print(f"Using device : {device}")

# ── Config ─────────────────────────────────────────────
DATA_DIR       = '//media/isesat/e8188905-1ffc-4de1-83b6-ac2addc2a941'   # path to your dataset folder
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic"}
IGNORE_FOLDERS = {'.Trash-1001', 'lost+found'}
RUN_EXTRACTION = False
NUM_CLASSES=100

# ── Load Dataset ───────────────────────────────────────
class FilteredImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [
            folder for folder in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, folder))
            and folder not in IGNORE_FOLDERS
        ]
        classes.sort()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx

dataset = FilteredImageFolder(
    root=DATA_DIR,
    transform=preprocess_val,  # ← using CLIP's own preprocessing
    is_valid_file=lambda path: os.path.splitext(path)[1].lower() in IMAGE_EXTS
)

print(f"Classes found : {len(dataset.classes)}")
print(f"Total images  : {len(dataset)}")


class SafeDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        try:
            return self.subset[idx]
        except Exception:
            return None
        
def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)
# Stratified split
indices = list(range(len(dataset)))
labels  = [dataset.targets[i] for i in indices]

train_val_idx, test_idx = train_test_split(
    indices, test_size=0.15, stratify=labels, random_state=42
)

train_val_labels = [labels[i] for i in train_val_idx]
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=0.17647, stratify=train_val_labels, random_state=42
)

train_dataset = Subset(dataset, train_idx)
test_dataset  = Subset(dataset, test_idx)

train_loader = DataLoader(SafeDataset(train_dataset), batch_size=128,
                          shuffle=False, num_workers=20, pin_memory=True,
                          collate_fn=collate_skip_none, prefetch_factor=2)
test_loader  = DataLoader(SafeDataset(test_dataset),  batch_size=128,
                          shuffle=False, num_workers=20, pin_memory=True,
                          collate_fn=collate_skip_none, prefetch_factor=2)

print(f"Train : {len(train_dataset)} | Test : {len(test_dataset)}")

def extract_clip_embeddings(loader, model, device):
    all_embeddings = []
    all_labels     = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            if batch is None:
                continue
            images, labels = batch
            images = images.to(device)

            # CLIP has a built in method for image embeddings
            embeddings = model.encode_image(images)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_embeddings, axis=0),
        np.concatenate(all_labels,     axis=0)
    )

if RUN_EXTRACTION:
    print("Extracting train embeddings ...")
    train_embeddings, train_labels = extract_clip_embeddings(train_loader, model, device)

    print("Extracting test embeddings ...")
    test_embeddings, test_labels = extract_clip_embeddings(test_loader, model, device)

    print(f"Train embeddings : {train_embeddings.shape}")
    print(f"Test embeddings  : {test_embeddings.shape}")

    np.savez('clip_embeddings.npz',
             train_embeddings = train_embeddings,
             train_labels     = train_labels,
             test_embeddings  = test_embeddings,
             test_labels      = test_labels)
    print("Embeddings saved → clip_embeddings.npz")

else:
    data             = np.load('clip_embeddings.npz')
    train_embeddings = data['train_embeddings']
    train_labels     = data['train_labels']
    test_embeddings  = data['test_embeddings']
    test_labels      = data['test_labels']
    print("Embeddings loaded from clip_embeddings.npz")

def build_faiss_index(embeddings):
    """Build a FAISS index from train embeddings."""
    embeddings = embeddings.astype('float32')
    
    # Normalise embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product = cosine similarity after normalisation
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def evaluate_retrieval(test_embeddings, test_labels, index, train_labels, k=5):
    query = test_embeddings.astype('float32')
    faiss.normalize_L2(query)

    p_at_1 = []
    p_at_k = []
    
    chunk_size = 1000  # search 1000 queries at a time

    for start in tqdm(range(0, len(query), chunk_size), desc="Precision Evaluation"):
        end        = min(start + chunk_size, len(query))
        chunk      = query[start:end]
        
        _, indices = index.search(chunk, k)
        
        for i, idx in enumerate(indices):
            query_label  = test_labels[start + i]
            top_k_labels = train_labels[idx]
            correct      = (top_k_labels == query_label)

            p_at_1.append(float(correct[0]))
            p_at_k.append(correct.sum() / k)

    print(f"Precision@1  : {np.mean(p_at_1):.4f}")
    print(f"Precision@{k} : {np.mean(p_at_k):.4f}")


def knn_evaluation(test_embeddings, test_labels, index, train_labels, k=21):
    query = test_embeddings.astype('float32')
    faiss.normalize_L2(query)

    correct    = 0
    chunk_size = 1000

    for start in tqdm(range(0, len(query), chunk_size), desc="kNN Evaluation"):
        end        = min(start + chunk_size, len(query))
        chunk      = query[start:end]

        _, indices = index.search(chunk, k)

        for i, idx in enumerate(indices):
            query_label     = test_labels[start + i]
            top_k_labels    = train_labels[idx]
            votes           = np.bincount(top_k_labels, minlength=NUM_CLASSES)
            predicted_class = np.argmax(votes)

            if predicted_class == query_label:
                correct += 1

    accuracy = correct / len(test_embeddings)
    print(f"kNN Accuracy (k={k}) : {accuracy:.4f}")

print("\n── Retrieval Evaluation ──")
index = build_faiss_index(train_embeddings)
#evaluate_retrieval(test_embeddings, test_labels, index, train_labels, k=5)
knn_evaluation(test_embeddings, test_labels, index, train_labels, k=21)