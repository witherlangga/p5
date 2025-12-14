import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rembg import remove
from sklearn.cluster import KMeans

# Klasifikasi warna centroid (RULE-BASED)
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

# Segmentasi daun dengan K-Means
def segment_leaf(image_path, K=10):

    # Remove background
    with open(image_path, "rb") as f:
        input_image = f.read()

    output_image = remove(input_image)
    result = Image.open(io.BytesIO(output_image)).convert("RGBA")

    # RGBA → RGB + Mask
    image_rgba = np.array(result)
    rgb = image_rgba[:, :, :3]
    alpha = image_rgba[:, :, 3]

    mask_leaf = alpha > 128
    pixels_leaf = np.float32(rgb[mask_leaf])

    # K-Means
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_leaf)
    centers = np.uint8(kmeans.cluster_centers_)

    labels_keterangan = [classify_leaf_color(c) for c in centers]

    # Gambar hasil segmentasi
    segmented_image = np.zeros_like(rgb)
    segmented_image[mask_leaf] = centers[labels]

    filename = os.path.splitext(os.path.basename(image_path))[0]
    result_path = f"static/results/seg_{filename}.png"
    Image.fromarray(segmented_image).save(result_path)

    # Data cluster
    height, width, _ = rgb.shape
    label_image = np.zeros((height, width), dtype=np.uint8)
    label_image[mask_leaf] = labels

    total_pixels = pixels_leaf.shape[0]

    cluster_results = []
    numeric_data = []

    bar_labels = []
    bar_values = []

    for i, desc in enumerate(labels_keterangan):
        cluster_index = i + 1

        cluster_mask = np.zeros((height, width, 3), dtype=np.uint8)
        cluster_mask[label_image == i] = centers[i]

        cluster_path = f"static/results/cluster_{cluster_index}_{filename}.png"
        Image.fromarray(cluster_mask).save(cluster_path)

        pixel_count = np.sum(labels == i)
        percentage = round((pixel_count / total_pixels) * 100, 2)

        cluster_results.append({
            "index": cluster_index,
            "description": desc,
            "path": cluster_path
        })

        numeric_data.append({
            "cluster": cluster_index,
            "centroid": f"({centers[i][0]}, {centers[i][1]}, {centers[i][2]})",
            "pixel_count": int(pixel_count),
            "percentage": percentage,
            "description": desc
        })

        bar_labels.append(f"Cluster {cluster_index}")
        bar_values.append(percentage)

    # Grafik BAR
    bar_path = f"static/results/bar_{filename}.png"

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bar_labels, bar_values)

    plt.title("Distribusi Persentase Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Persentase (%)")

    for bar, value in zip(bars, bar_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}%",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # Return ke Flask
    return {
        "result_path": result_path,
        "clusters": cluster_results,
        "numeric_data": numeric_data,
        "bar_chart": bar_path
    }
