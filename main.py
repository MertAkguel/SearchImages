import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

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


# ------------------------------------------------------------------
# Cache the model so it's loaded only once
# ------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolo11m.pt")  # Replace with your model path


# ------------------------------------------------------------------
# Detection function
# ------------------------------------------------------------------
def detect_with_yolo(image, selected_classes, conf: float = 0.25):
    model = load_yolo_model()
    all_class_names = model.names
    name_to_index = {v: k for k, v in all_class_names.items()}
    target_labels = [name_to_index[name] for name in selected_classes if name in name_to_index]

    results = model.predict(image, classes=target_labels, conf=conf)
    return results[0] if results else None


# ------------------------------------------------------------------
# Main Streamlit application
# ------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Image Class Finder",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Sidebar - class selection
    with st.sidebar:
        st.header("Class Filter Settings")
        selected_classes = st.multiselect(
            "Choose target class(es)",
            options=COCO_CLASSES,
            placeholder="Select one or more classes",
        )
        selected_conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    # Main area
    st.title("📤 Upload Images for Class Filtering")
    uploaded_files = st.file_uploader(
        "Upload image files (JPG, PNG, BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True
    )

    if st.button("Run Detection", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one image.")
            return

        if not selected_classes:
            st.error("Please select at least one class from the sidebar.")
            return

        st.info(f"Processing {len(uploaded_files)} image(s)...")
        matched_images = []

        progress = st.progress(0)
        status = st.empty()

        for i, file in enumerate(uploaded_files, start=1):
            image = Image.open(BytesIO(file.read())).convert("RGB")
            result = detect_with_yolo(image, selected_classes, selected_conf)

            if result and result.boxes and len(result.boxes.cls) > 0:
                matched_images.append((file.name, image))

            progress.progress(i / len(uploaded_files))
            status.text(f"{i}/{len(uploaded_files)} processed...")

        st.success(f"Detection complete. {len(matched_images)} match(es) found.")
        if matched_images:
            st.subheader("🖼️ Matched Images")
            for filename, img in matched_images:
                st.image(img, caption=filename, use_container_width=True)


if __name__ == "__main__":
    main()
