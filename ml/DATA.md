# How to get the required datasets

The ML training scripts expect the following data. All paths are relative to the `ml/` directory.

---

## 1. Scene classifier: CIFAR-10

**No manual download.** The first time you run the classifier training script, torchvision will download CIFAR-10 automatically into `data/cifar10/`.

```bash
cd ml
python training/train_classifier.py   # downloads CIFAR-10 on first run
```

---

## 2. Saliency: SALICON

SALICON uses images and saliency (fixation) maps from the SALICON / LSUN challenge (based on MS COCO). You need to download the archives and place files so the layout matches what the training script expects.

### Step 1: Download (from [salicon.net/challenge](https://www.salicon.net/challenge/))

| Content | Size | Link |
|--------|------|------|
| **Images** (train + val) | 3.0 GB | [Google Drive: Images](https://drive.google.com/uc?id=1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5&export=download) |
| **Fixation maps** (saliency maps) | 0.4 GB | [Google Drive: Fixation Maps](https://drive.google.com/uc?id=1PnO7szbdub1559LfjYHMy65EDC4VhJC8&export=download) |

(Optional: Fixations raw data 1.4 GB — [link](https://drive.google.com/uc?id=1P-jeZXCsjoKO79OhFUgnj6FGcyvmLDPj&export=download) — not required if you use the fixation maps.)

### Step 2: Extract and arrange

The training script expects this layout under `ml/data/salicon/`:

```
data/salicon/
  images/
    train/   ← training images (e.g. COCO_train2014_*.jpg)
    val/     ← validation images (e.g. COCO_val2014_*.jpg)
  maps/
    train/   ← saliency maps for training (same base names as images, e.g. .png)
    val/     ← saliency maps for validation
```

- Extract the **Images** zip. It usually contains folders or names that indicate train vs validation (e.g. `train2014/`, `val2014/` or filenames like `COCO_train2014_*.jpg` and `COCO_val2014_*.jpg`). Copy or symlink so that:
  - Training images end up in `data/salicon/images/train/`
  - Validation images end up in `data/salicon/images/val/`
- Extract the **Fixation maps** zip. You get saliency maps (e.g. PNG per image). Split or copy them so that:
  - Maps for training images are in `data/salicon/maps/train/`
  - Maps for validation images are in `data/salicon/maps/val/`
- Ensure **matching filenames**: for each image in `images/train/` there must be a map in `maps/train/` with the same base name (e.g. `COCO_train2014_000000001.jpg` → `COCO_train2014_000000001.png`). The code looks for the same stem + extension or `.png` in the maps folder.

If the downloaded zips use different naming or layout, add a small script or manual step to copy files into the above structure. Then run:

```bash
cd ml
python training/train_saliency.py
```

---

## 3. Super-resolution: DIV2K

DIV2K is the standard benchmark for super-resolution. You only need the **high-resolution (HR)** images for train and validation; the training code generates low-res on the fly.

### Step 1: Download (from [data.vision.ee.ethz.ch/cvl/DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/))

| Content | Link |
|--------|------|
| **Train HR** (800 images) | [DIV2K_train_HR.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) |
| **Validation HR** (100 images) | [DIV2K_valid_HR.zip](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) |

### Step 2: Extract

Extract both zips so that the **folder names** are exactly:

- `DIV2K_train_HR`  (contains `0001.png`, `0002.png`, …)
- `DIV2K_valid_HR`  (contains `0801.png`, …)

Place them inside `ml/data/div2k/`. Final layout:

```
ml/data/div2k/
  DIV2K_train_HR/   ← 0001.png, 0002.png, ..., 0800.png
  DIV2K_valid_HR/   ← 0801.png, ..., 0900.png
```

Example (from project root):

```bash
cd "Real-Time-Camera-Processing-Pipeline/ml/data"
mkdir -p div2k
cd div2k
# After downloading the two zips to this folder:
unzip DIV2K_train_HR.zip
unzip DIV2K_valid_HR.zip
# If the zip extracts to a different name, rename to DIV2K_train_HR and DIV2K_valid_HR
```

Then run:

```bash
cd ml
python training/train_superres.py
```

---

## Reorganizing after download

If your SALICON or DIV2K downloads use different folder names or nesting, run the organizer script so the layout matches what the training code expects:

```bash
# From project root
python ml/scripts/organize_data.py
```

This script:

- **DIV2K:** Moves PNGs from `DIV2K_valid_HR/DIV2K_valid_HR/` (and same for train if present) up into `DIV2K_valid_HR/` and `DIV2K_train_HR/`.
- **SALICON:** Moves fixation maps from `maps/map/train` and `maps/map/val` into `maps/train` and `maps/val`, and copies images from `img/train` into `images/train` if that gives you more images.

**SALICON validation images:** The training script expects both `images/train` and `images/val`. If `images/val` is empty, put the validation images (e.g. `COCO_val2014_*.jpg` from the SALICON Images zip) into `data/salicon/images/val/` so they match the filenames in `maps/val/`. If you only have train images, saliency training will run only on train; you can add a small val split or leave val empty and fix the dataset class to allow empty val.

## Quick reference

| Model        | Data path           | Get it |
|-------------|---------------------|--------|
| Classifier  | `data/cifar10`      | Auto-download when running `training/train_classifier.py` |
| Saliency    | `data/salicon/images/{train,val}`, `maps/{train,val}` | Download Images + Fixation Maps from salicon.net/challenge, then arrange as above |
| Super-res   | `data/div2k/DIV2K_train_HR`, `DIV2K_valid_HR` | Download train/valid HR zips from data.vision.ee.ethz.ch/cvl/DIV2K and extract into `data/div2k/` |

After all three are in place, run training then export:

```bash
cd ml
python training/train_classifier.py
python training/train_saliency.py
python training/train_superres.py
python export/export_classifier.py
python export/export_saliency.py
python export/export_superres.py
```

Use `WANDB_MODE=disabled` if you don’t use Weights & Biases, e.g.:

`WANDB_MODE=disabled python training/train_classifier.py`
