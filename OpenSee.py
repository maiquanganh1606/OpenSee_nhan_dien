import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import shutil
import json
import av
import random
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- CẤU HÌNH ---
DATASET_PATH = 'dataset'
TRAINER_PATH = 'trainer/trainer.yml'
NAMES_FILE = 'names.json'
LIKES_FILE = 'likes.json'
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# --- KHỞI TẠO FILE ---
if not os.path.exists(DATASET_PATH): os.makedirs(DATASET_PATH)
if not os.path.exists('trainer'): os.makedirs('trainer')
if not os.path.exists(NAMES_FILE):
    with open(NAMES_FILE, 'w') as f: json.dump({"0": "Unknown"}, f)
if not os.path.exists(LIKES_FILE):
    with open(LIKES_FILE, 'w') as f: json.dump({"count": 0}, f)


# --- UTILS ---
def load_names():
    with open(NAMES_FILE, 'r') as f: return json.load(f)


def save_name_to_json(names):
    with open(NAMES_FILE, 'w') as f: json.dump(names, f)


def get_likes():
    try:
        with open(LIKES_FILE, 'r') as f:
            return json.load(f)['count']
    except:
        return 0


def add_like():
    current = get_likes()
    new_count = current + 1
    with open(LIKES_FILE, 'w') as f: json.dump({"count": new_count}, f)
    return new_count


def get_new_id():
    names = load_names()
    current_ids = [int(k) for k in names.keys()]
    return max(current_ids) + 1 if current_ids else 1


def get_face_detector():
    return cv2.CascadeClassifier(CASCADE_PATH)


# ==========================================
# CLASS QUAY VIDEO (PHẦN 1)
# ==========================================
class FaceCollector(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = get_face_detector()
        self.save_mode = False
        self.face_id = -1
        self.count = 0
        self.frame_skip = 0

    def set_params(self, save_mode, face_id):
        self.save_mode = save_mode
        self.face_id = face_id

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Soi gương

        if self.save_mode and self.face_id != -1:
            self.frame_skip += 1
            if self.frame_skip % 5 == 0:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    self.count += 1
                    file_name = f"User.{self.face_id}.{self.count}.jpg"
                    cv2.imwrite(os.path.join(DATASET_PATH, file_name), gray[y:y + h, x:x + w])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, f"SAVING {self.count}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return img


# --- GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="Face ID Pro Max", layout="wide", page_icon="💖")
st.title("🤖 OpenSee - Nhận Diện Và Chấm Điểm Nhan Sắc")

# CSS Fix Camera Lật (Giữ nguyên để phần 3 soi gương được)
st.markdown("""<style>div[data-testid="stCameraInput"] video {transform: scaleX(-1) !important;}</style>""",
            unsafe_allow_html=True)

# ==========================================
# SIDEBAR (TƯƠNG TÁC ĐƠN GIẢN)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
st.sidebar.title("🤖 OpenSee Xin Chào ")
st.sidebar.title("Menu Chức năng")
menu = ["1. Thu thập dữ liệu", "2. Quản lý & Huấn luyện", "3. Nhận diện (Chụp ảnh)"]
choice = st.sidebar.selectbox("Chọn:", menu)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💌 Góc Tương Tác")
st.sidebar.write("**Nếu bạn thấy OpenSee giỏi hãy tặng 1 tym nhé!**")

col_heart1, col_heart2 = st.sidebar.columns([1, 2])
current_likes = get_likes()

# Logic nút bấm Tim (Đã xóa hiệu ứng mưa)
with col_heart1:
    if st.button("❤️"):
        add_like()  # Cộng 1 tim
        st.toast("Cảm ơn bạn đã thả tym! ❤️", icon="🥰")  # Thông báo nhẹ nhàng
        st.rerun()  # Cập nhật số hiển thị ngay

with col_heart2:
    st.metric(label="Số lượng Tym", value=current_likes)

st.sidebar.markdown("---")
st.sidebar.info("Developed by **Quang Anh**")

# ==========================================
# NỘI DUNG CHÍNH
# ==========================================

# --- 1. THU THẬP ---
if choice == "1. Thu thập dữ liệu":
    st.header("📸 Thu thập dữ liệu")
    col1, col2 = st.columns(2)
    with col1:
        name_input = st.text_input("Tên người mới (hiện tại Tiếng Việt còn đang bị lỗi, nhập không dấu bạn nhé)  /nVD:Nguyen Van A):")
    if 'new_id' not in st.session_state: st.session_state.new_id = get_new_id()

    if name_input:
        st.info(f"ID cấp: **{st.session_state.new_id}** - **{name_input}**")
        src = st.radio("Nguồn:", ["🔴 Quay Live", "📁 Upload Video"])

        if src == "🔴 Quay Live":
            rec = st.checkbox("GHI HÌNH (REC)", value=False)
            ctx = webrtc_streamer(key="collect", video_processor_factory=FaceCollector)
            if ctx.video_processor: ctx.video_processor.set_params(rec, st.session_state.new_id)
        else:
            up_vid = st.file_uploader("Chọn video", type=['mp4'])
            if up_vid and st.button("Trích xuất"):
                tfile = open("temp_video.mp4", "wb")
                tfile.write(up_vid.read())
                vidcap = cv2.VideoCapture("temp_video.mp4")
                count, saved_count = 0, 0
                st_img = st.empty()
                detector = get_face_detector()
                while True:
                    success, frame = vidcap.read()
                    if not success: break
                    frame = cv2.flip(frame, 1)
                    if count % 5 == 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = detector.detectMultiScale(gray, 1.1, 4)
                        for (x, y, w, h) in faces:
                            saved_count += 1
                            f_name = f"User.{st.session_state.new_id}.{saved_count}.jpg"
                            cv2.imwrite(os.path.join(DATASET_PATH, f_name), gray[y:y + h, x:x + w])
                    count += 1
                st.success(f"Đã lưu {saved_count} ảnh!")

        if st.button("💾 Lưu Người Dùng"):
            names = load_names()
            names[str(st.session_state.new_id)] = name_input
            save_name_to_json(names)
            st.success("Đã lưu!")
            st.session_state.new_id = get_new_id()

# --- 2. QUẢN LÝ ---
elif choice == "2. Quản lý & Huấn luyện":
    st.header("🛠️ Quản lý & Huấn luyện")
    names = load_names()
    tab1, tab2, tab3, tab4 = st.tabs(["✏️ Sửa Tên", "🗑️ Dọn Ảnh Chưa Đẹp", "❌ Xóa Người Dùng", "🧠 Huấn Luyện OpenSee"])

    # TAB 1: SỬA TÊN
    with tab1:
        id_ed = st.selectbox("Chọn ID:", list(names.keys()), format_func=lambda x: f"{x}: {names[x]}")
        new_n = st.text_input("Tên mới:", value=names[id_ed])
        if st.button("Cập nhật"):
            names[id_ed] = new_n
            save_name_to_json(names)
            st.rerun()

    # TAB 2: DỌN ẢNH (TÍNH NĂNG MỚI: XEM ẢNH)
    with tab2:
        id_clean = st.selectbox("Chọn ID dọn dẹp:", list(names.keys()), format_func=lambda x: f"{x}: {names[x]}",
                                key='clean')
        files = [f for f in os.listdir(DATASET_PATH) if f.startswith(f"User.{id_clean}.")]
        st.write(f"Tìm thấy {len(files)} ảnh trong dữ liệu.")

        # [NEW] TÍNH NĂNG XEM ẢNH TRƯỚC KHI XÓA
        if len(files) > 0:
            with st.expander("👁️ Bấm vào đây để XEM TOÀN BỘ ẢNH của người này"):
                st.info("Mẹo: Nhìn tên file bên dưới ảnh (VD: User.1.25.jpg) để chọn xóa cho chính xác.")
                # Tạo lưới 5 cột
                cols = st.columns(5)
                for idx, file_name in enumerate(files):
                    img_path = os.path.join(DATASET_PATH, file_name)
                    try:
                        image = Image.open(img_path)
                        with cols[idx % 5]:
                            # Hiển thị ảnh và tên file làm caption
                            st.image(image, caption=file_name, use_column_width=True)
                    except:
                        continue
        # ------------------------------------------------

        del_imgs = st.multiselect("Chọn ảnh xấu/mờ để xóa:", files)

        if st.button("🗑️ Xóa ảnh đã chọn"):
            if len(del_imgs) > 0:
                for f in del_imgs:
                    os.remove(os.path.join(DATASET_PATH, f))
                st.success(f"Đã xóa {len(del_imgs)} ảnh!")
                st.rerun()
            else:
                st.warning("Bạn chưa chọn ảnh nào để xóa.")

    # TAB 3: XÓA NGƯỜI DÙNG
    with tab3:
        st.warning("⚠️ CẢNH BÁO: Xóa ID sẽ xóa luôn toàn bộ ảnh trong dataset của người đó.")
        id_del = st.selectbox("Chọn Người muốn xóa VĨNH VIỄN:", list(names.keys()),
                              format_func=lambda x: f"ID {x}: {names[x]}", key='delete_user')
        if id_del == "0":
            st.info("Không thể xóa ID 0 (Quang Anh).")
        else:
            if st.button(f"🔴 Xác nhận XÓA {names[id_del]}"):
                all_files = os.listdir(DATASET_PATH)
                for f in all_files:
                    if f.startswith(f"User.{id_del}."): os.remove(os.path.join(DATASET_PATH, f))
                del names[id_del]
                save_name_to_json(names)
                st.success(f"Đã xóa!")
                st.rerun()

    # TAB 4: HUẤN LUYỆN
    with tab4:
        st.header('🤖  Huấn luyện để OpenSee làm quen thêm người mới nào')
        if st.button("🚀 Train Model"):
            rec = cv2.face.LBPHFaceRecognizer_create()
            det = get_face_detector()
            samps, ids = [], []
            files = os.listdir(DATASET_PATH)
            bar = st.progress(0)
            for i, f in enumerate(files):
                try:
                    p = os.path.join(DATASET_PATH, f)
                    im = Image.open(p).convert('L')
                    np_im = np.array(im, 'uint8')
                    id = int(f.split('.')[1])
                    faces = det.detectMultiScale(np_im)
                    for (x, y, w, h) in faces:
                        samps.append(np_im[y:y + h, x:x + w])
                        ids.append(id)
                    bar.progress((i + 1) / len(files))
                except:
                    pass
            if ids:
                rec.train(samps, np.array(ids))
                rec.write(TRAINER_PATH)
                st.success(f"Xong! 🤖 OpenSee đã biết {len(np.unique(ids))} người.")
            else:
                st.error("Không có dữ liệu!")

# --- 3. NHẬN DIỆN (CHỤP ẢNH) ---
elif choice == "3. Nhận diện (Chụp ảnh)":
    st.header("🕵️ Nhận diện & Chấm Điểm Nhan sắc")

    if not os.path.exists(TRAINER_PATH):
        st.error("⚠️ Chưa có Model! Hãy Train trước.")
    else:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_PATH)
        face_cascade = get_face_detector()
        names = load_names()

        img_input = st.camera_input("Bấm nút chụp ảnh")

        if img_input:
            image = Image.open(img_input)
            img_arr = np.array(image.convert('RGB'))

            # [QUAN TRỌNG: FIX LỖI KHUNG]
            # 1. Lật ảnh ngay lập tức để khớp với mắt người dùng
            img_arr = cv2.flip(img_arr, 1)

            # 2. Tạo bản copy để xử lý nhận diện (Gray)
            # Lưu ý: OpenCV dùng BGR, Streamlit dùng RGB.
            # Convert từ RGB sang Gray
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

            # 3. Detect khuôn mặt với tham số chặt chẽ hơn để tránh nhiễu
            # scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),  # [FIX] Loại bỏ khung nhiễu quá nhỏ
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # 4. Vẽ khung lên ảnh GỐC (img_arr - đang là RGB)
            if len(faces) == 0:
                st.warning("Ảnh không rõ chụp lại nhé!")
            elif len(faces)>0:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    try:
                        id, conf = recognizer.predict(roi_gray)
                        if conf < 100:
                            name = names.get(str(id), "Unknown")
                            color = (0, 255, 0)  # Xanh lá (RGB)
                        else:
                            name = "Unknown"
                            color = (255, 0, 0)  # Đỏ (RGB)

                        if name != "Unknown":
                            beauty_score = random.choice(["9/10", "10/10", "Sieu Pham!"])
                            display_text = f"{name} - {beauty_score}"
                        else:
                            display_text = 'Unknown'

                        # Vẽ hình chữ nhật (RGB)
                        cv2.rectangle(img_arr, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img_arr, display_text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except:
                        pass
                st.image(img_arr, caption="Kết quả", use_container_width=True)
                if name!='Unknown':
                    st.success("😳😵‍💫😍  OpenSee đã bị quyến rũ bởi nhan sắc này 🤖")
                else:
                    st.warning("🤔 OpenSee chưa nhận ra bạn!  \nHãy quay lại Menu 1 để thu thập dữ liệu, sau đó qua Menu 2 Train Model để OpenSee nhận ra bạn nhé! 😊")
