import os
import io
import numpy as np
from PIL import Image
from rembg import remove
from sklearn.cluster import KMeans


def classify_leaf_color(rgb_color):
    r, g, b = rgb_color

    if g > r and g > b:
        return "Daun Sehat (Hijau)"
    elif r > 150 and g > 150 and b < 100:
        return "Daun Sedikit Sakit (Kuning)"
    elif r > g and r > b and r > 120:
        return "Daun Sakit (Coklat Kekuningan)"
    elif r < 90 and g < 90 and b < 90:
        return "Daun Rusak Parah (Gelap)"
    else:
        return "Lainnya"


def segment_leaf(image_path):

    # ===== Tahap 2.5: Remove background =====
    with open(image_path, "rb") as f:
        input_image = f.read()

    output_image = remove(input_image)
    result = Image.open(io.BytesIO(output_image)).convert("RGBA")

    # ===== Tahap 3: RGBA → RGB + mask =====
    image_rgba = np.array(result)
    rgb = image_rgba[:, :, :3]
    alpha = image_rgba[:, :, 3]

    mask_leaf = alpha > 128
    pixels_leaf = np.float32(rgb[mask_leaf])

    # ===== Tahap 4: K-Means =====
    K = 10
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_leaf)
    centers = np.uint8(kmeans.cluster_centers_)

    # ===== Tahap 4.5: Klasifikasi =====
    labels_keterangan = [classify_leaf_color(c) for c in centers]

    # ===== Tahap 5: Rekonstruksi full segmentation =====
    segmented_image = np.zeros_like(rgb)
    segmented_image[mask_leaf] = centers[labels]

    filename = os.path.splitext(os.path.basename(image_path))[0]

    result_path = f"static/results/seg_{filename}.png"
    Image.fromarray(segmented_image).save(result_path)

    # ===== Tahap 6: SIMPAN SETIAP CLUSTER (INI YANG HILANG) =====
    height, width, _ = rgb.shape
    label_image = np.zeros((height, width), dtype=np.uint8)
    label_image[mask_leaf] = labels

    cluster_results = []

    for i, desc in enumerate(labels_keterangan):
        cluster_mask = np.zeros((height, width, 3), dtype=np.uint8)
        cluster_mask[label_image == i] = centers[i]

        cluster_path = f"static/results/cluster_{i+1}_{filename}.png"
        Image.fromarray(cluster_mask).save(cluster_path)

        cluster_results.append({
            "index": i + 1,
            "description": desc,
            "path": cluster_path
        })

    return {
        "result_path": result_path,
        "clusters": cluster_results
    }
