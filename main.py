from src.dataset import LFDataset
from time import time
from PIL import Image
import os
from segmentor import Segmentor


def main():
    dataset = LFDataset("/home/ngoncharov/cvpr2026/datasets/ycbv_lf/mustard0")
    s_size, t_size = dataset.metadata["n_views"]
    segmentor = Segmentor(prompt="bottle.")
    os.makedirs("cutie_output", exist_ok=True)
    time_now = time()
    for i, frame in enumerate(dataset):
        img_central = frame["LF"][s_size // 2, t_size // 2]

        mask = segmentor(img_central)

        Image.fromarray((mask.cpu().numpy() * 255).astype("uint8")).save(
            f"cutie_output/frame_{i:04d}.png"
        )

    time_per_frame = (time() - time_now) / len(dataset)
    print(time_per_frame)


if __name__ == "__main__":
    main()
