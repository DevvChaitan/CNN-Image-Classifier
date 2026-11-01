"""CIFAR-10 CNN classifier and Streamlit demo.

Install the runtime packages with:
    python -m pip install tensorflow streamlit pillow py7zr matplotlib

Run the app with:
    streamlit run CIFAR10_CNN_Streamlit_Project.py

For a fresh model and report plots, run:
    python CIFAR10_CNN_Streamlit_Project.py --train --epochs 8

The repository includes a small trained model so the public app starts without
training. For a fresh training run, train.7z and trainLabels.csv are used when
they are next to the repository; otherwise Keras downloads canonical CIFAR-10.
Generated training outputs are written to the system temp folder.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
RANDOM_SEED = 42
DEFAULT_EPOCHS = 8
BATCH_SIZE = 128
MAX_TRAIN_PER_CLASS = 1_000
IMAGE_SIZE = (32, 32)


def runtime_directory() -> Path:
    """Return a writable cache directory outside the submitted project."""
    configured = os.environ.get("CIFAR10_CACHE_DIR")
    path = Path(configured) if configured else Path(tempfile.gettempdir()) / "cifar10_cnn_streamlit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_local_dataset() -> tuple[Path, Path] | None:
    """Find the provided Kaggle-format dataset beside this repository."""
    candidates = [
        Path(os.environ["CIFAR10_DATA_DIR"]) if os.environ.get("CIFAR10_DATA_DIR") else None,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        archive = candidate / "train.7z"
        labels = candidate / "trainLabels.csv"
        if archive.is_file() and labels.is_file():
            return archive, labels
    return None


def stratified_indices(labels: np.ndarray, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split labeled examples 80/10/10 while retaining each class balance."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for class_id in range(len(CLASS_NAMES)):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        train_end = int(len(indices) * 0.8)
        validation_end = int(len(indices) * 0.9)
        train_parts.append(indices[:train_end])
        validation_parts.append(indices[train_end:validation_end])
        test_parts.append(indices[validation_end:])
    train = np.concatenate(train_parts)
    validation = np.concatenate(validation_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def load_local_kaggle_data(archive_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Extract and load the 50,000 labeled PNG images provided with the project."""
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError("Install py7zr to load the provided train.7z archive.") from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install Pillow to read the CIFAR-10 PNG files.") from exc

    extracted_root = runtime_directory() / "provided_dataset"
    images_dir = extracted_root / "train"
    marker = images_dir / "50000.png"
    if not marker.is_file():
        extracted_root.mkdir(parents=True, exist_ok=True)
        tar_command = shutil.which("tar")
        extracted_with_tar = False
        if tar_command:
            try:
                subprocess.run(
                    [tar_command, "-xf", str(archive_path), "-C", str(extracted_root)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                extracted_with_tar = True
            except subprocess.CalledProcessError:
                # Some tar builds cannot read 7z; py7zr is the portable fallback.
                pass
        if not extracted_with_tar:
            with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                archive.extractall(path=extracted_root)

    records: list[tuple[int, int]] = []
    class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    with labels_path.open("r", newline="", encoding="utf-8-sig") as label_file:
        for row in csv.DictReader(label_file):
            try:
                image_id = int(row["id"])
                class_id = class_to_id[row["label"].strip().lower()]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Unexpected row in {labels_path.name}: {row}") from exc
            records.append((image_id, class_id))

    if len(records) != 50_000:
        raise ValueError(f"Expected 50,000 labeled images; found {len(records):,}.")

    images = np.empty((len(records), 32, 32, 3), dtype=np.uint8)
    labels = np.empty((len(records),), dtype=np.int64)
    for index, (image_id, class_id) in enumerate(records):
        image_path = images_dir / f"{image_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image referenced by labels: {image_path}")
        with Image.open(image_path) as image:
            images[index] = np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.uint8)
        labels[index] = class_id
    return images, labels, "the provided 50,000-image training set"


def load_canonical_cifar10() -> tuple[np.ndarray, np.ndarray, str]:
    """Load CIFAR-10 from Keras when the local Kaggle files are not present."""
    import tensorflow as tf

    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    return images, labels.reshape(-1).astype(np.int64), "the canonical CIFAR-10 training set"


def load_dataset() -> tuple[np.ndarray, np.ndarray, str]:
    local_files = find_local_dataset()
    if local_files:
        return load_local_kaggle_data(*local_files)
    return load_canonical_cifar10()


def build_cnn() -> Any:
    """Create a compact CNN for 32x32 RGB images and 10 classes."""
    import tensorflow as tf

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
        ],
        name="data_augmentation",
    )
    inputs = tf.keras.Input(shape=(32, 32, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    for filters, dropout_rate in ((32, 0.15), (64, 0.20)):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_compact_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(epochs: int = DEFAULT_EPOCHS, verbose: int = 2) -> tuple[Any, dict[str, Any]]:
    """Train the CNN and evaluate it on a stratified held-out test split."""
    import tensorflow as tf

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    images, labels, source_name = load_dataset()
    training_pool_ids, validation_ids, test_ids = stratified_indices(labels)
    rng = np.random.default_rng(RANDOM_SEED)
    selected_train_parts: list[np.ndarray] = []
    for class_id in range(len(CLASS_NAMES)):
        class_train_ids = training_pool_ids[labels[training_pool_ids] == class_id].copy()
        rng.shuffle(class_train_ids)
        selected_train_parts.append(class_train_ids[:MAX_TRAIN_PER_CLASS])
    train_ids = np.concatenate(selected_train_parts)
    rng.shuffle(train_ids)
    x_train, y_train = images[train_ids], labels[train_ids]
    x_validation, y_validation = images[validation_ids], labels[validation_ids]
    x_test, y_test = images[test_ids], labels[test_ids]

    model = build_cnn()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
            ),
        ],
        verbose=verbose,
    )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    predicted_ids = np.argmax(predictions, axis=1)
    correct_ids = np.flatnonzero(y_test == predicted_ids)
    incorrect_ids = np.flatnonzero(y_test != predicted_ids)
    example_ids: list[int] = []
    if len(correct_ids):
        example_ids.append(int(correct_ids[0]))
    if len(incorrect_ids):
        example_ids.append(int(incorrect_ids[0]))
    example_ids.extend(index for index in range(len(y_test)) if index not in example_ids)
    example_ids = example_ids[:10]
    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)

    metrics = {
        "source": source_name,
        "split_counts": {
            "train": int(len(train_ids)),
            "available_training_pool": int(len(training_pool_ids)),
            "validation": int(len(validation_ids)),
            "test": int(len(test_ids)),
        },
        "epochs_completed": len(history.history["loss"]),
        "training_accuracy": float(history.history["accuracy"][-1]),
        "validation_accuracy": float(history.history["val_accuracy"][-1]),
        "training_loss": float(history.history["loss"][-1]),
        "validation_loss": float(history.history["val_loss"][-1]),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "class_names": CLASS_NAMES,
        "model_summary": "\n".join(summary_lines),
        "test_labels": y_test.astype(int).tolist(),
        "test_predictions": predicted_ids.astype(int).tolist(),
        "test_probabilities": predictions.astype(float).tolist(),
        "test_example_indices": example_ids,
        "test_examples": x_test[example_ids].astype(np.uint8).tolist(),
    }
    return model, metrics


def save_training_outputs(model: Any, metrics: dict[str, Any], output_dir: Path) -> None:
    """Save temporary model and figures used to prepare the project report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "cifar10_cnn.keras")
    compact_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"test_probabilities", "test_examples"}
    }
    (output_dir / "metrics.json").write_text(json.dumps(compact_metrics, indent=2), encoding="utf-8")

    history = metrics["history"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(history["accuracy"], label="Training")
    axes[0].plot(history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(history["loss"], label="Training")
    axes[1].plot(history["val_loss"], label="Validation")
    axes[1].set_title("Loss by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary_lines = metrics["model_summary"].splitlines()
    figure_height = max(6.5, len(summary_lines) * 0.23)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    axis.axis("off")
    axis.text(0.02, 0.98, "\n".join(summary_lines), va="top", family="monospace", fontsize=8.2)
    figure.tight_layout()
    figure.savefig(output_dir / "model_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    labels = np.asarray(metrics["test_labels"], dtype=np.int64)
    predictions = np.asarray(metrics["test_predictions"], dtype=np.int64)
    probabilities = np.asarray(metrics["test_probabilities"], dtype=np.float32)
    selected = metrics["test_example_indices"]
    examples = np.asarray(metrics["test_examples"], dtype=np.uint8)

    figure, axes = plt.subplots(2, 5, figsize=(11, 5.8))
    for axis, index, example in zip(axes.flat, selected, examples):
        axis.imshow(example)
        was_correct = labels[index] == predictions[index]
        axis.set_title(
            f"Actual: {CLASS_NAMES[labels[index]]}\nPred: {CLASS_NAMES[predictions[index]]} "
            f"({probabilities[index, predictions[index]]:.0%})",
            color="#16834b" if was_correct else "#bd3c3c",
            fontsize=8,
        )
        axis.axis("off")
    figure.suptitle("Held-out test predictions (green = correct, red = incorrect)")
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_examples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_cifar10_model() -> tuple[Any, str, dict[str, Any] | None]:
    """Load pre-trained model for low-latency web inference without retraining."""
    import tensorflow as tf

    repo_dir = Path(__file__).resolve().parent
    bundled_model_path = repo_dir / "cifar10_cnn.keras"
    bundled_weights_path = repo_dir / "cifar10_cnn.weights.h5"
    cache_model_path = runtime_directory() / "cifar10_cnn.keras"

    # Attempt 1: Direct load_model with compile=False (fastest, prevents optimizer conflicts)
    for candidate in (bundled_model_path, cache_model_path):
        if candidate.is_file():
            try:
                model = tf.keras.models.load_model(candidate, compile=False)
                return model, "bundle" if candidate == bundled_model_path else "cache", None
            except Exception:
                pass

    # Attempt 2: Reconstruct exact CNN architecture and load weights
    for candidate in (bundled_weights_path, bundled_model_path, cache_model_path):
        if candidate.is_file():
            try:
                model = build_cnn()
                model.load_weights(candidate)
                return model, "weights", None
            except Exception:
                pass

    # Attempt 3: Standard load_model
    if bundled_model_path.is_file():
        try:
            model = tf.keras.models.load_model(bundled_model_path)
            return model, "bundle", None
        except Exception:
            pass

    # Do NOT trigger automatic CPU training on Streamlit Cloud
    raise FileNotFoundError(
        "Pre-trained model files ('cifar10_cnn.keras' or 'cifar10_cnn.weights.h5') could not be loaded. "
        "Please check repository weights or run training locally via: python CIFAR10_CNN_Streamlit_Project.py --train"
    )


def streamlit_app() -> None:
    """Render the image upload and classification experience."""
    import streamlit as st
    from PIL import Image, UnidentifiedImageError

    st.set_page_config(
        page_title="CIFAR-10 Image Classifier",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.title("🧠 CIFAR-10 Image Classifier")
    st.write(
        "Upload a small photo or choose a sample image below to see real-time CNN predictions. "
        "The model classifies across 10 CIFAR-10 categories: **airplane**, **automobile**, **bird**, "
        "**cat**, **deer**, **dog**, **frog**, **horse**, **ship**, and **truck**."
    )
    with st.expander("ℹ️ How it works & Model Architecture"):
        st.markdown(
            "1. **Input Preprocessing**: The image is converted to RGB and resized to 32 × 32 pixels.\n"
            "2. **Convolutional Neural Network**: 2 convolutional blocks (32 & 64 filters) with max pooling and dropout.\n"
            "3. **Dense Classification**: Global average pooling feeding into a 96-unit dense layer with softmax activation.\n"
            "4. **Class Probabilities**: The highest model confidence score is shown alongside the full class probability breakdown."
        )
        st.caption("Displayed values represent model confidence scores on 32×32 inputs.")

    # Cache model resource in memory across Streamlit reruns
    @st.cache_resource(show_spinner="Loading pre-trained CNN model…")
    def _get_cached_model():
        try:
            return load_cifar10_model()
        except Exception as exc:
            return None, "error", {"error": str(exc)}

    model, source_type, info = _get_cached_model()
    if source_type == "error":
        st.error(f"❌ Failed to load model: {info.get('error', 'Unknown error') if info else 'Unknown error'}")
        st.info("Ensure pre-trained model weights (cifar10_cnn.keras or cifar10_cnn.weights.h5) are present in the repo.")
        return
    elif source_type in ("bundle", "weights"):
        st.caption("✅ Pre-trained CNN weights loaded successfully.")
    else:
        st.caption("⚡ Model loaded from runtime cache.")

    # Sample images option for zero-friction testing
    sample_dir = Path(__file__).resolve().parent / "sample_images"
    sample_files = sorted(sample_dir.glob("*.png")) if sample_dir.is_dir() else []

    selected_image: Image.Image | None = None
    image_source_label = ""

    if sample_files:
        sample_map = {f.stem.replace("_sample", "").title(): f for f in sample_files}
        source_mode = st.radio(
            "Select input method:",
            ["Choose from sample images", "Upload your own image"],
            horizontal=True,
        )

        if source_mode == "Choose from sample images":
            sample_choice = st.selectbox(
                "Pick a sample image to test:",
                list(sample_map.keys()),
            )
            if sample_choice:
                sample_path = sample_map[sample_choice]
                try:
                    with Image.open(sample_path) as src:
                        selected_image = src.convert("RGB")
                    image_source_label = f"Sample image: {sample_choice}"
                except Exception as exc:
                    st.error(f"Could not load sample image: {exc}")
        else:
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["png", "jpg", "jpeg", "webp"],
                help="Upload a PNG, JPG, JPEG, or WebP image",
            )
            if uploaded_file is not None:
                try:
                    with Image.open(uploaded_file) as source_image:
                        selected_image = source_image.convert("RGB")
                    image_source_label = f"Uploaded: {uploaded_file.name}"
                except (UnidentifiedImageError, OSError, ValueError):
                    st.error("That file could not be read as an image. Upload a valid PNG, JPG, JPEG, or WebP file.")
                    return
    else:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a PNG, JPG, JPEG, or WebP image",
        )
        if uploaded_file is not None:
            try:
                with Image.open(uploaded_file) as source_image:
                    selected_image = source_image.convert("RGB")
                image_source_label = f"Uploaded: {uploaded_file.name}"
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("That file could not be read as an image. Upload a valid PNG, JPG, JPEG, or WebP file.")
                return

    if selected_image is None:
        st.info("👆 Upload an image or select a sample image above to see the CNN prediction.")
        return

    left, right = st.columns([1, 1])
    with left:
        preview = selected_image.resize((256, 256), Image.Resampling.NEAREST)
        preview_buffer = io.BytesIO()
        preview.save(preview_buffer, format="PNG")
        preview_data = base64.b64encode(preview_buffer.getvalue()).decode("ascii")
        st.markdown(
            f'<img src="data:image/png;base64,{preview_data}" alt="Image preview" '
            'style="width:100%;max-width:256px;border-radius:10px;image-rendering:pixelated;box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">',
            unsafe_allow_html=True,
        )
        st.caption(f"{image_source_label} (resized to 32×32 for CNN)")

    model_input = np.asarray(selected_image.resize(IMAGE_SIZE), dtype=np.uint8)[np.newaxis, ...]
    scores = model.predict(model_input, verbose=0)[0]
    predicted_id = int(np.argmax(scores))
    confidence = float(scores[predicted_id])

    with right:
        st.subheader("Prediction")
        st.metric(
            label="Most Likely Class",
            value=CLASS_NAMES[predicted_id].upper(),
            delta=f"{confidence:.1%} confidence",
        )
        st.progress(min(max(confidence, 0.0), 1.0))

    st.subheader("Confidence Scores across all 10 Classes")
    score_table = {name.title(): float(scores[index]) for index, name in enumerate(CLASS_NAMES)}
    st.bar_chart(score_table, horizontal=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN project and Streamlit app")
    parser.add_argument("--train", action="store_true", help="train a model and save report plots")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"maximum epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=runtime_directory(),
        help="where runtime model and report plots are saved (default: system temp folder)",
    )
    args, _unknown = parser.parse_known_args()
    if args.train:
        if args.epochs < 1:
            parser.error("--epochs must be at least 1")
        model, metrics = train_model(epochs=args.epochs)
        save_training_outputs(model, metrics, args.output_dir)
        omitted = {
            "history",
            "model_summary",
            "test_labels",
            "test_predictions",
            "test_probabilities",
            "test_examples",
            "test_example_indices",
        }
        print(json.dumps({key: value for key, value in metrics.items() if key not in omitted}, indent=2))
        print(f"\nTemporary training outputs saved to: {args.output_dir}")
    else:
        streamlit_app()


if __name__ == "__main__":
    main()
