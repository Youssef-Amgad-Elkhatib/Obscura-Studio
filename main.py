import streamlit as st
import cv2 as cv
import numpy as np
import os
import hashlib
from PIL import Image
from streamlit_cropper import st_cropper
from ultralytics import YOLO

# ==========================================
# 1. OPENCV BACKEND FUNCTIONS
# ==========================================

@st.cache_resource
def load_model():
    return YOLO("yolov11n-face.pt")


def detect_and_filter_face(
    img, yolo_model, filter_type=None, img_path=None,
    kernel=None, sigma=None, colors=None, blocks=None
):
    if img is None or img.ndim != 3:
        return img

    results = yolo_model.predict(img, conf=0.4, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            face_roi = img[y1:y2, x1:x2]

            if filter_type == "emoji":
                img[y1:y2, x1:x2] = replace_with_emoji(face_roi, img_path)
            elif filter_type == "blur":
                img[y1:y2, x1:x2] = blur_face(face_roi, kernel, sigma)
            elif filter_type == "color":
                img[y1:y2, x1:x2] = color_face(face_roi, colors)
            elif filter_type == "pixelate":
                img[y1:y2, x1:x2] = pixelate_face(face_roi, blocks)
            elif filter_type == "censor":
                img[y1:y2, x1:x2] = draw_censor_bar(face_roi)

    return img


def blur_face(face_roi, kernel, sigma):
    return cv.GaussianBlur(face_roi, kernel, sigma)


def replace_with_emoji(face_roi, img_path):
    emoji_img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    if emoji_img is None:
        return face_roi

    h, w = face_roi.shape[:2]
    emoji_img = cv.resize(emoji_img, (w, h), interpolation=cv.INTER_AREA)
    if emoji_img.shape[2] == 4:
        b, g, r, a = cv.split(emoji_img)
        emoji_bgr = cv.merge((b, g, r))
        alpha_mask = a / 255.0
        alpha_mask = np.expand_dims(alpha_mask, axis=2)
        blended = (emoji_bgr * alpha_mask) + (face_roi * (1.0 - alpha_mask))
        return blended.astype(np.uint8)
    else:
        return emoji_img


def blur_img(img, d, sigma_color, sigma_space):
    return cv.bilateralFilter(img, d, sigma_color, sigma_space)


def color_face(face_roi, colors_bgr):
    new_color_face_roi = np.zeros_like(face_roi)
    new_color_face_roi[:] = colors_bgr
    return new_color_face_roi


def pixelate_face(face_roi, blocks=15):
    h, w = face_roi.shape[:2]
    tiny_face = cv.resize(face_roi, (blocks, blocks), interpolation=cv.INTER_LINEAR)
    return cv.resize(tiny_face, (w, h), interpolation=cv.INTER_NEAREST)


def draw_censor_bar(face_roi):
    h, w = face_roi.shape[:2]
    start_y = int(h / 3)
    end_y = int(start_y + (h / 3.5))
    face_roi[start_y:end_y, 0:w] = (0, 0, 0)
    return face_roi


def flip_img(img, flip_code):
    return cv.flip(img, flip_code)


def rotate_img(img, angle, rot_point=None):
    height, width = img.shape[:2]
    if rot_point is None:
        rot_point = (width // 2, height // 2)

    rot_mat = cv.getRotationMatrix2D(rot_point, angle, 1.0)
    return cv.warpAffine(img, rot_mat, (width, height))


# ==========================================
# 2. COLOR / FILE HELPERS
# ==========================================

def hex_to_bgr(hex_color):
    """Convert CSS #RRGGBB -> OpenCV (B, G, R)."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])


def get_emoji_files():
    if not os.path.exists("emojis"):
        return []

    return sorted(
        os.path.join("emojis", f)
        for f in os.listdir("emojis")
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )


def read_uploaded_image(uploaded_file):
    """Decode the uploaded file exactly once as BGR."""
    file_bytes = uploaded_file.getvalue()
    img = cv.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv.IMREAD_COLOR)

    if img is None:
        raise ValueError("OpenCV could not decode this image.")

    return img, hashlib.sha256(file_bytes).hexdigest()


# ==========================================
# 3. UI INJECTION
# ==========================================

def inject_custom_css():
    st.markdown(
        """
        <style>
            .stApp { background: #F5F7FB; color: #172033; }
            [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E3E8F0; }
            h1, h2, h3 { color: #172033; }

            div.stButton > button:first-child,
            div.stDownloadButton > button:first-child {
                border-radius: 10px; font-weight: 650; min-height: 42px; transition: all 0.2s ease;
            }
            div.stButton > button:first-child {
                background: #263A5B; color: white; border: 1px solid #263A5B;
            }
            div.stButton > button:first-child:hover {
                background: #344E78; border-color: #344E78;
            }
            div.stDownloadButton > button:first-child {
                background: #C8922E; color: white; border: 1px solid #C8922E;
            }
            div.stDownloadButton > button:first-child:hover {
                background: #A97720; border-color: #A97720;
            }
            .image-card {
                background: white; padding: 14px; border-radius: 14px;
                border: 1px solid #E3E8F0; box-shadow: 0 4px 18px rgba(23, 32, 51, 0.06);
                text-align: center;
            }
            .image-card img { border-radius: 8px; }
            .section-title {
                color: #263A5B; font-size: 16px; font-weight: 700; margin-bottom: 8px;
            }
            .spacer { margin-top: 20px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# 4. MAIN APP
# ==========================================

def main():
    st.set_page_config(
        page_title="Obscura Studio",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    st.title("Obscura Studio 📸")
    st.markdown("Upload one image, edit it, undo/redo changes, and download the result.")

    haar_cascade = load_model()

    # ------------------------------------------
    # SESSION STATE
    # ------------------------------------------
    if "history" not in st.session_state:
        st.session_state.history = []

    if "current_step" not in st.session_state:
        st.session_state.current_step = -1

    if "uploaded_hash" not in st.session_state:
        st.session_state.uploaded_hash = None

    if "selected_emoji" not in st.session_state:
        st.session_state.selected_emoji = None

    # ------------------------------------------
    # SINGLE IMAGE UPLOAD
    # ------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        if st.session_state.uploaded_hash is not None:
            # The user removed the uploader's file.
            st.session_state.history = []
            st.session_state.current_step = -1
            st.session_state.uploaded_hash = None
            st.session_state.selected_emoji = None
        st.info("Upload an image to start editing.")
        return

    try:
        original_img, upload_hash = read_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    # IMPORTANT: A different upload completely replaces the previous image and history.
    if upload_hash != st.session_state.uploaded_hash:
        st.session_state.history = [original_img.copy()]
        st.session_state.current_step = 0
        st.session_state.uploaded_hash = upload_hash
        st.session_state.selected_emoji = None

    current_img = st.session_state.history[st.session_state.current_step]

    # ------------------------------------------
    # SIDEBAR
    # ------------------------------------------
    st.sidebar.header("🔧 Toolkit")

    action = st.sidebar.selectbox(
        "Choose an Action",
        [
            "None",
            "Crop",
            "Redact Faces",
            "Global Blur",
            "Rotate",
            "Flip",
            "Convert Color Space",
        ],
    )

    preview_img = current_img.copy()
    commit = False

    # ------------------------------------------
    # FACE / BLUR OPERATIONS
    # ------------------------------------------
    if action == "Redact Faces":
        st.sidebar.subheader("Face Redaction")

        filter_type = st.sidebar.selectbox(
            "Filter",
            ["blur", "pixelate", "censor", "color", "emoji"],
        )

        kernel_size = None
        sigma = None
        colors = None
        blocks = None
        selected_emoji = None

        if filter_type == "blur":
            k = st.sidebar.slider("Blur Kernel Size", 11, 199, 99, step=2)
            kernel_size = (k, k)
            sigma = st.sidebar.slider("Sigma", 0, 100, 30)

        elif filter_type == "pixelate":
            blocks = st.sidebar.slider("Pixel Block Size", 5, 50, 15)

        elif filter_type == "color":
            hex_color = st.sidebar.color_picker("Pick a color", "#000000")
            colors = hex_to_bgr(hex_color)

        elif filter_type == "emoji":
            emojis = get_emoji_files()

            if not emojis:
                st.sidebar.error("Create an 'emojis' folder and put PNG/JPG/WebP images in it.")
            else:
                st.sidebar.markdown('<div class="section-title">Choose an Emoji</div>', unsafe_allow_html=True)

                emoji_cols = st.sidebar.columns(3)
                for i, emoji_path in enumerate(emojis):
                    with emoji_cols[i % 3]:
                        emoji_preview = Image.open(emoji_path).convert("RGBA")
                        emoji_preview = emoji_preview.resize((150, 150))
                        st.image(emoji_preview, use_container_width=True)

                        if st.button(
                                "✓" if st.session_state.selected_emoji == emoji_path else "Select",
                                key=f"emoji_{i}_{os.path.basename(emoji_path)}",
                                use_container_width=True,
                        ):
                            st.session_state.selected_emoji = emoji_path
                            st.rerun()

                selected_emoji = st.session_state.selected_emoji

        if st.sidebar.button("Apply Face Redaction", use_container_width=True):
            if filter_type == "emoji" and selected_emoji is None:
                st.sidebar.warning("Select an emoji first.")
            else:
                preview_img = detect_and_filter_face(
                    preview_img,
                    haar_cascade,
                    filter_type=filter_type,
                    img_path=selected_emoji,
                    kernel=kernel_size,
                    sigma=sigma,
                    colors=colors,
                    blocks=blocks,
                )
                commit = True

    elif action == "Global Blur":
        st.sidebar.subheader("Bilateral Blur")

        d = st.sidebar.slider("Diameter (d)", 1, 30, 9)
        sigma_c = st.sidebar.slider("Sigma Color", 1, 150, 75)
        sigma_s = st.sidebar.slider("Sigma Space", 1, 150, 75)

        if st.sidebar.button("Apply Global Blur", use_container_width=True):
            preview_img = blur_img(preview_img, d, sigma_c, sigma_s)
            commit = True

    # ------------------------------------------
    # SPATIAL OPERATIONS
    # ------------------------------------------
    elif action == "Rotate":
        st.sidebar.subheader("Rotation")
        angle = st.sidebar.slider("Angle", -180, 180, 0)

        if st.sidebar.button("Apply Rotation", use_container_width=True):
            preview_img = rotate_img(preview_img, angle)
            commit = True

    elif action == "Flip":
        st.sidebar.subheader("Flip")
        flip_dir = st.sidebar.radio("Direction", ["Horizontal", "Vertical", "Both"])
        flip_code = 1 if flip_dir == "Horizontal" else 0 if flip_dir == "Vertical" else -1

        if st.sidebar.button("Apply Flip", use_container_width=True):
            preview_img = flip_img(preview_img, flip_code)
            commit = True

    # ------------------------------------------
    # COLOR SPACE CONVERSION
    # ------------------------------------------
    elif action == "Convert Color Space":
        st.sidebar.subheader("Color Space")

        # Now functions as a permanent destructive filter just like a blur
        target_space = st.sidebar.selectbox(
            "Convert to",
            ["Grayscale", "HSV", "RGB"],
        )

        if st.sidebar.button("Apply Color Space", use_container_width=True):
            if target_space == "Grayscale":
                gray = cv.cvtColor(preview_img, cv.COLOR_BGR2GRAY)
                # We convert it back to BGR purely so it remains a 3-channel array.
                # This ensures you can still add colored emojis or censor bars to it!
                preview_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            elif target_space == "HSV":
                preview_img = cv.cvtColor(preview_img, cv.COLOR_BGR2HSV)
            elif target_space == "RGB":
                preview_img = cv.cvtColor(preview_img, cv.COLOR_BGR2RGB)

            commit = True

    # ------------------------------------------
    # COMMIT
    # ------------------------------------------
    if commit:
        # Remove redo states before adding a new edit.
        if st.session_state.current_step < len(st.session_state.history) - 1:
            st.session_state.history = st.session_state.history[: st.session_state.current_step + 1]

        st.session_state.history.append(preview_img.copy())
        st.session_state.current_step += 1
        st.rerun()

    # ------------------------------------------
    # MAIN IMAGE DISPLAY
    # ------------------------------------------
    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:
        # We blindly treat the internal array as BGR to force the funky
        # color space visualizations to show up correctly in Streamlit.
        display_img = cv.cvtColor(current_img, cv.COLOR_BGR2RGB)

        if action == "Crop":
            st.info("🖌️ Draw a box over the image, then click 'Apply Crop' in the sidebar.")
            pil_img = Image.fromarray(display_img)
            cropped_pil = st_cropper(
                pil_img,
                realtime_update=True,
                box_color="#C8922E",
                aspect_ratio=None,
            )

            if st.sidebar.button("Apply Crop", use_container_width=True):
                crop_rgb = np.array(cropped_pil)
                preview_img = cv.cvtColor(crop_rgb, cv.COLOR_RGB2BGR)

                st.session_state.history = st.session_state.history[: st.session_state.current_step + 1]
                st.session_state.history.append(preview_img.copy())
                st.session_state.current_step += 1
                st.rerun()
        else:
            st.image(display_img, use_container_width=True)
            st.markdown(f"**Edit Step: {st.session_state.current_step}**")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # --------------------------------------
        # CENTERED UNDO / REDO
        # --------------------------------------
        spacer_left, btn_undo, btn_redo, spacer_right = st.columns([1.5, 1, 1, 1.5])

        with btn_undo:
            undo_disabled = st.session_state.current_step <= 0
            if st.button("↶ Undo", disabled=undo_disabled, use_container_width=True):
                st.session_state.current_step -= 1
                st.rerun()

        with btn_redo:
            redo_disabled = st.session_state.current_step >= len(st.session_state.history) - 1
            if st.button("Redo ↷", disabled=redo_disabled, use_container_width=True):
                st.session_state.current_step += 1
                st.rerun()

        # --------------------------------------
        # CENTERED DOWNLOAD
        # --------------------------------------
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        dl_left, dl_center, dl_right = st.columns([1.5, 2, 1.5])

        with dl_center:
            is_success, buffer = cv.imencode(".jpg", current_img, [cv.IMWRITE_JPEG_QUALITY, 95])
            if is_success:
                st.download_button(
                    label="⬇ Download Image",
                    data=buffer.tobytes(),
                    file_name="edited_image.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
