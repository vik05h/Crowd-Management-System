import glob
import os
import random
import cv2

def render_sample_annotations(num_samples: int = 3):
    img_dir = "dataset/images/train"
    lbl_dir = "dataset/labels/train"
    out_dir = "dataset/verification_samples"
    os.makedirs(out_dir, exist_ok=True)

    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    if not img_files:
        print("[ERROR] No images found in dataset/images/train")
        return

    random.seed(42)
    selected = random.sample(img_files, min(num_samples, len(img_files)))

    for img_path in selected:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{basename}.txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = int((xc - bw / 2.0) * w)
                    y1 = int((yc - bh / 2.0) * h)
                    x2 = int((xc + bw / 2.0) * w)
                    y2 = int((yc + bh / 2.0) * h)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        "person",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            out_path = os.path.join(out_dir, f"verified_{basename[:15]}.jpg")
            cv2.imwrite(out_path, img)
            print(f"[VERIFIED] Saved visual sample: {out_path} ({len(lines)} annotations)")

if __name__ == "__main__":
    render_sample_annotations(3)
