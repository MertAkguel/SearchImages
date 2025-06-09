import streamlit as st
from pathlib import Path
from ultralytics import YOLO

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# -----------------------------------------------------------------------------
# Dummy detection functions (replace with real inference later)
# -----------------------------------------------------------------------------

def detect_with_yolo(image, selected_classes, conf: int = 0.25):
    # Load model
    model = YOLO("yolo11m.pt")  # replace with your model path

    # Map class names to indices
    all_class_names = model.names  # dict: {0: 'person', 1: 'bicycle', ...}
    name_to_index = {v: k for k, v in all_class_names.items()}

    # Convert selected class names to indices
    target_labels = [name_to_index[name] for name in selected_classes if name in name_to_index]

    # Now call predict with integer class indices
    results = model.predict(image, classes=target_labels, conf=conf)
    return True if len(results[0].boxes.cls) > 0 is not None else False


# -----------------------------------------------------------------------------
# Main Streamlit application
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Image Class Finder",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # ===== Sidebar – Model & Class Selection =================================
    with st.sidebar:
        st.header("Model & Class Selection")
        model_choice = st.radio("Select model", ["YOLOv11"], index=0)

        if model_choice == "YOLOv11":
            selected_classes = st.multiselect(
                "Choose target class(es)",
                options=COCO_CLASSES,
                placeholder="Select one or more classes",
            )
            selected_conf = st.slider("Select a Conf Value", 0.0, 1.0, 0.5)
        else:
            class_text = st.text_input(
                "Enter target class name(s)",
                placeholder="e.g. zebra, golden retriever",
            )
            selected_classes = [c.strip() for c in class_text.split(",") if c.strip()]

        st.markdown("---")
        st.write("After choosing the directory in the main panel, click **Run** to start.")

    # ===== Main Panel – Directory Input & Run Button ==========================
    st.title("🔍 Search Images by Class")

    dir_path_str = st.text_input(
        "Directory containing images",
        placeholder="/path/to/your/images",
        label_visibility="visible",
    )

    run_button = st.button("Run", type="primary")

    # =====Processing ==============================================================
    if run_button:
        # --- Input validation ---------------------------------------------------
        if not dir_path_str:
            st.error("Please enter a directory path before running.")
            st.stop()

        directory = Path(dir_path_str)
        if not directory.exists() or not directory.is_dir():
            st.error("The provided path is not a valid directory.")
            st.stop()

        if not selected_classes:
            st.error("Please select or enter at least one class label in the sidebar.")
            st.stop()

        # Gather candidate images ------------------------------------------------
        image_files = [
            p for p in directory.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        total_images = len(image_files)

        if total_images == 0:
            st.warning("No image files found in the specified directory.")
            st.stop()

        st.info(f"Found **{total_images}** images. Beginning analysis…")

        progress_bar = st.progress(0, text="Starting…")
        status_placeholder = st.empty()
        found_images: list[Path] = []

        # Spinner while processing ---------------------------------------------
        with st.spinner("Processing images – please wait…"):
            for idx, img_path in enumerate(image_files, start=1):
                # Choose detection function based on model selection
                if model_choice == "YOLOv11":
                    detected = detect_with_yolo(img_path, selected_classes, selected_conf)

                if detected:
                    found_images.append(img_path)

                # Update progress -------------------------------------------------------------------
                progress_bar.progress(idx / total_images, text=f"{idx}/{total_images} images processed")
                status_placeholder.metric(
                    label="Matches found so far",
                    value=str(len(found_images)),
                    delta=None,
                )

        # ===== Results =========================================================
        st.success(
            f"Completed! Processed **{total_images}** images **{len(found_images)}** match(es) found."
        )

        if found_images:
            st.subheader("Matched Images")
            for img_path in found_images:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
