import torch
import numpy as np
import os
import json
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from streamlit_image_coordinates import streamlit_image_coordinates

# You can change these if you want
CHEKPOINT = "checkpoints/sam2.1_hiera_base_plus.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
INPUT_DIR = "test_images"

# Set page to wide mode
st.set_page_config(layout="wide")

# Set wide sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 500px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'predictor' not in st.session_state:
 
    # Load SAM2 model
    checkpoint = CHEKPOINT
    model_cfg = MODEL_CFG

    @st.cache_resource
    def load_model():
        model = build_sam2(model_cfg, checkpoint, 
                           mask_threshold=0.75, 
                           max_hole_area=1,
                           max_sprinkle_area=1
                          )
        return SAM2ImagePredictor(model)

    st.session_state.predictor = load_model()

if 'current_image' not in st.session_state:
    st.session_state.current_image = None

if 'current_masks' not in st.session_state:
    st.session_state.current_masks = None

if 'points' not in st.session_state:
    st.session_state.points = []

if 'point_labels' not in st.session_state:
    st.session_state.point_labels = []

if 'bbox' not in st.session_state:
    st.session_state.bbox = None

if 'drawing_mode' not in st.session_state:
    st.session_state.drawing_mode = "point"  # or "box"

if 'current_image_idx' not in st.session_state:
    st.session_state.current_image_idx = 0



def show_mask(mask, random_color=False):

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)

    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Convert to uint8 for OpenCV operations
    mask_image = (mask_image * 255).astype(np.uint8)

    # Find contours
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]

    # Create a copy for drawing contours
    mask_with_contours = mask_image.copy()

    # Draw contours
    cv2.drawContours(mask_with_contours, contours, -1, (255, 255, 255, 128), thickness=2)

    # Convert back to float32 for matplotlib
    return mask_with_contours.astype(np.float32) / 255.0


 
def save_results(image, masks, filename):

    # Save masks as JSON
    masks_list = []
    for i, mask in enumerate(masks):
        mask_dict = {
            "segmentation": mask.tolist(),
            "area": float(mask.sum()),
            "bbox": cv2.boundingRect(mask.astype(np.uint8)),
            "id": i
        }
        masks_list.append(mask_dict)

    json_filename = os.path.join("output", f"{filename}_masks.json")

    with open(json_filename, 'w') as f:
        json.dump(masks_list, f)

    # Save visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for mask in masks:
        plt.imshow(show_mask(mask), alpha=0.5)

    plt.axis('off')
    plt.savefig(os.path.join("output", f"{filename}_visualization.png"), 

                bbox_inches='tight', pad_inches=0)
    plt.close()


 
def draw_points_and_box_on_image(image, points=None, bbox=None):
 
    # Convert PIL image to numpy array
    img_array = np.array(image)
 
    # Create a copy of the image to draw on
    img_with_points = img_array.copy()
 
    # Draw points if they exist
    if points and len(points) > 0:
        points = np.array(points)
 
        for x, y in points:
            cv2.circle(img_with_points, (int(x), int(y)), 5, (255, 0, 0), -1)  # Red dots
            cv2.circle(img_with_points, (int(x), int(y)), 5, (255, 255, 255), 1)  # White border
 
    # Draw bbox if it exists
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox[0])
        cv2.rectangle(img_with_points, (x1, y1), (x2, y2), (255, 0, 0), 2)
 
    return img_with_points
 
st.title("SAM2 Image Segmentation")
 
# Get list of images
image_files = sorted(os.listdir(INPUT_DIR))
total_images = len(image_files)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    st.write(f"Current photo: {st.session_state.current_image_idx + 1}/{total_images}")
 
    # Image navigation
    col1, col2= st.columns(2)
 
    with col1:
 
        if st.button("⬅️ Previous photo", use_container_width=True):
            st.session_state.current_image_idx = (st.session_state.current_image_idx - 1) % total_images
            st.rerun()
 
    with col2:
        if st.button("Next photo ➡️", use_container_width=True):
            st.session_state.current_image_idx = (st.session_state.current_image_idx + 1) % total_images
            st.rerun()
 
    # Image selection
    selected_image = st.selectbox(
        "Select an image",
        image_files,
        index=st.session_state.current_image_idx,
        key="image_selector"
    )
 
    # Update current_image_idx when selection changes
    st.session_state.current_image_idx = image_files.index(selected_image)
 
    # Drawing mode selection
    mode = st.radio(
        "Drawing Mode",
        ["point", "box"],
        format_func=lambda x: "Point Selection" if x == "point" else "Box Selection",
        key="mode_selection"
    )
 
    st.session_state.drawing_mode = mode
 
    # Point prompts
    if st.button("Clear All"):
        st.session_state.points = []
        st.session_state.point_labels = []
        st.session_state.bbox = None
        st.rerun()
 
    # Segmentation controls
    st.subheader("Segmentation")
 
    # Run segmentation
    if st.button("Run Segmentation", use_container_width=True):
 
        with st.spinner("Running segmentation..."):
            point_coords = np.array(st.session_state.points) if st.session_state.points else None
            point_labels = np.array(st.session_state.point_labels) if st.session_state.points else None
 
            masks, scores, _ = st.session_state.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=st.session_state.bbox,
                multimask_output=False
            )
 
            st.session_state.current_masks = masks
 
    # Automatic segmentation
    if st.button("Run Automatic Segmentation", use_container_width=True):
 
        image_path = os.path.join(INPUT_DIR, selected_image)
        current_image = Image.open(image_path)
 
        with st.spinner("Running automatic segmentation..."):
            mask_generator = SAM2AutomaticMaskGenerator(st.session_state.predictor.model)
            # Convert PIL image to numpy array
            img_array = np.array(current_image)
            masks = mask_generator.generate(img_array)
            st.session_state.current_masks = [mask["segmentation"] for mask in masks]
 
    
    # Save results
    if st.session_state.current_masks is not None:
 
        if st.button("Save Results", use_container_width=True):
            filename = f"{os.path.splitext(selected_image)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            image_path = os.path.join(INPUT_DIR, selected_image)
            current_image = Image.open(image_path)
            save_results(np.array(current_image), st.session_state.current_masks, filename)
            st.success(f"Results saved to output/{filename}")
 
# Main content area
if selected_image:
 
    image_path = os.path.join(INPUT_DIR, selected_image)
 
    if st.session_state.current_image is None or image_path != st.session_state.current_image:
        image = Image.open(image_path)
        st.session_state.current_image = image_path
        st.session_state.predictor.set_image(image)
        st.session_state.points = []
        st.session_state.point_labels = []
        st.session_state.bbox = None
        st.session_state.current_masks = None
        
    image = Image.open(image_path)
    col1, col2 = st.columns([0.5,0.5])
    
    with col1:
        st.subheader("Original Image")
        # Draw points and box on image
        img_with_annotations = draw_points_and_box_on_image(
            image, 
            st.session_state.points, 
            st.session_state.bbox
        )
 
        # Display image with click coordinates
        value = streamlit_image_coordinates(
            img_with_annotations,
            key="coordinates"
        )
 
        
        # Handle click events
        if value is not None and value != st.session_state.get('last_click'):
 
            st.session_state['last_click'] = value
            x, y = value['x'], value['y']
            if st.session_state.drawing_mode == "point":
                st.session_state.points.append([x, y])
                st.session_state.point_labels.append(1)  # Foreground point
                st.rerun()  # Rerun to update the image with new point
 
            elif st.session_state.drawing_mode == "box" and len(st.session_state.points) < 2:
                st.session_state.points.append([x, y])
                if len(st.session_state.points) == 2:
                    p1, p2 = st.session_state.points
                    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                    st.session_state.bbox = np.array([[x1, y1, x2, y2]])
                    st.session_state.points = []  # Clear points after creating bbox
 
                st.rerun()  # Rerun to update the image with new box
 
    
    with col2:
        # Display results below the main image if available
        if st.session_state.current_masks is not None:
            st.subheader("Results")
            fig, ax = plt.subplots(figsize=(10,10))
            ax.imshow(image)
            for mask in st.session_state.current_masks:
                ax.imshow(show_mask(mask))#, alpha=0.8)
            ax.axis('off')
            st.pyplot(fig)


 
# Add click handler
if st.session_state.drawing_mode == "point":
    st.markdown("""
        <script>
        const img = window.parent.document.querySelector('img');
        img.style.cursor = 'crosshair';
        img.onclick = function(e) {
            const rect = e.target.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            window.parent.stStreamlitRerun({
                last_clicked_pos: [x, y]
            });
        };
        </script>
        """, unsafe_allow_html=True) 
