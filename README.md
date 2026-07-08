# Reflection-Robust 6-DoF Object Tracking in Light Fields

Each frame is lifted
to a relightable surface light field, separated into a diffuse view and a
reflected environment map, and tracked with LoFTR/ICP plus photometric pose
refinement.

## 🛠️ Setup

```bash
git clone --recursive <repo-url>
cd ReLiFT-6DoF
docker build -t relift6dof .
```

Download the LoFTR outdoor weights (`outdoor_ds.ckpt`) from the
[LoFTR repository](https://github.com/zju3dv/LoFTR) and place them in
`LoFTR/weights/outdoor_ds.ckpt`.

## 📦 Dataset

Download [SpecTrack](https://huggingface.co/datasets/nagongh/SpecTrack):

```bash
wget https://huggingface.co/datasets/nagongh/SpecTrack/resolve/main/dataset.tar.gz
tar -xzf dataset.tar.gz
```

## 🚀 Run

```bash
docker run --gpus all -it --rm --network host \
  -v $(pwd):/workspace \
  -v /path/to/SpecTrack_dataset:/data \
  relift6dof

python main.py --data /data --out results
```

One `(N, 4, 4)` object-to-camera trajectory (`.npy`) is written per sequence.

| Option | |
|---|---|
| `--data` | dataset root or a single sequence directory |
| `--out` | output directory (default `results`) |
| `--depth gt` | use dataset depth maps instead of light-field depth estimation |
| `--vis` | watch the photometric refinement live at http://localhost:8080 |

## ⚙️ Configuration

All hyperparameters are in [`config.yaml`](config.yaml).
