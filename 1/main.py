import io
import os
import traceback
import numpy as np
import streamlit as st
import cv2
from PIL import Image

st.set_page_config(page_title="Online OpenCV Editor", layout="wide")

# --- Optional: drawable canvas for mask drawing ---
try:
    from streamlit import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False


# -------------------- Helpers --------------------
SUPPORTED_EXT = {"jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"}

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """PIL -> OpenCV BGR uint8"""
    if pil_img.mode in ("RGBA", "LA"):
        pil_img = pil_img.convert("RGBA")
        arr = np.array(pil_img)
        # RGBA -> BGR (drop alpha)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return bgr
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def resize_image(img: np.ndarray, mode: str, scale: float, out_w: int, out_h: int, interp_choice: str):
    h, w = img.shape[:2]

    if mode == "Scale":
        if scale <= 0:
            raise ValueError("Коэффициент масштабирования должен быть > 0.")
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
    else:
        if out_w <= 0 or out_h <= 0:
            raise ValueError("Ширина и высота должны быть > 0.")
        new_w, new_h = out_w, out_h

    # Интерполяция
    interp_map = {
        "Nearest": cv2.INTER_NEAREST,
        "Bilinear": cv2.INTER_LINEAR,
        "Bicubic": cv2.INTER_CUBIC,
        "Auto": None,
    }

    if interp_choice != "Auto":
        interp = interp_map[interp_choice]
    else:
        # Автовыбор: если сильно увеличиваем — bicubic; если уменьшаем — area; иначе linear
        scale_x = new_w / w
        scale_y = new_h / h
        s = max(scale_x, scale_y)

        if s >= 1.5:
            interp = cv2.INTER_CUBIC
        elif s <= 0.8:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR

    out = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return out

def convert_from_bgr(img_bgr: np.ndarray, space: str) -> np.ndarray:
    """Convert BGR image to target color space representation."""
    space = space.upper()
    if space == "BGR":
        return img_bgr
    if space == "RGB":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if space == "GRAY":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if space == "HSV":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if space == "LAB":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    if space == "YCRCB":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    raise ValueError(f"Неизвестное цветовое пространство: {space}")

def convert_to_bgr(img_any: np.ndarray, space: str) -> np.ndarray:
    """Convert image from a given space back to BGR for display/encoding."""
    space = space.upper()
    if space == "BGR":
        return img_any
    if space == "RGB":
        return cv2.cvtColor(img_any, cv2.COLOR_RGB2BGR)
    if space == "GRAY":
        return cv2.cvtColor(img_any, cv2.COLOR_GRAY2BGR)
    if space == "HSV":
        return cv2.cvtColor(img_any, cv2.COLOR_HSV2BGR)
    if space == "LAB":
        return cv2.cvtColor(img_any, cv2.COLOR_LAB2BGR)
    if space == "YCRCB":
        return cv2.cvtColor(img_any, cv2.COLOR_YCrCb2BGR)
    raise ValueError(f"Неизвестное цветовое пространство: {space}")

def _hsv_mask_with_wrap(hsv: np.ndarray, h_lo: int, h_hi: int, s_lo: int, s_hi: int, v_lo: int, v_hi: int) -> np.ndarray:
    """
    HSV in OpenCV: H 0..179, S 0..255, V 0..255
    Handle hue wrap-around (e.g., near 0/179).
    """
    h_lo = int(clamp(h_lo, 0, 179))
    h_hi = int(clamp(h_hi, 0, 179))
    s_lo = int(clamp(s_lo, 0, 255)); s_hi = int(clamp(s_hi, 0, 255))
    v_lo = int(clamp(v_lo, 0, 255)); v_hi = int(clamp(v_hi, 0, 255))

    if h_lo <= h_hi:
        lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    # wrap: [h_lo..179] U [0..h_hi]
    lower1 = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper1 = np.array([179, s_hi, v_hi], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)

    lower2 = np.array([0, s_lo, v_lo], dtype=np.uint8)
    upper2 = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    m2 = cv2.inRange(hsv, lower2, upper2)

    return cv2.bitwise_or(m1, m2)

def find_object_by_color(
    img_bgr: np.ndarray,
    input_space: str,
    target_rgb: tuple[int, int, int] | None,
    target_hsv: tuple[int, int, int] | None,
    tol_h: int,
    tol_s: int,
    tol_v: int,
    tol_rgb: int,
    min_area: int,
    morph_open_k: int,
    morph_close_k: int,
):
    """
    Returns: (mask uint8 0/255, bbox (x,y,w,h) or None)
    """
    input_space = input_space.upper()

    if input_space == "RGB":
        if target_rgb is None:
            raise ValueError("target_rgb не задан.")
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        r, g, b = map(int, target_rgb)
        t = int(clamp(tol_rgb, 0, 255))
        lower = np.array([clamp(r - t, 0, 255), clamp(g - t, 0, 255), clamp(b - t, 0, 255)], dtype=np.uint8)
        upper = np.array([clamp(r + t, 0, 255), clamp(g + t, 0, 255), clamp(b + t, 0, 255)], dtype=np.uint8)
        mask = cv2.inRange(rgb, lower, upper)

    elif input_space == "HSV":
        if target_hsv is None:
            raise ValueError("target_hsv не задан.")
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = map(int, target_hsv)

        h_lo = (h - int(tol_h)) % 180
        h_hi = (h + int(tol_h)) % 180
        s_lo = s - int(tol_s); s_hi = s + int(tol_s)
        v_lo = v - int(tol_v); v_hi = v + int(tol_v)

        mask = _hsv_mask_with_wrap(hsv, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)

    else:
        raise ValueError("Поиск по цвету поддерживает только RGB или HSV.")

    # Морфология для очистки маски
    if morph_open_k and morph_open_k > 1:
        if morph_open_k % 2 == 0:
            morph_open_k += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_k, morph_open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if morph_close_k and morph_close_k > 1:
        if morph_close_k % 2 == 0:
            morph_close_k += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_k, morph_close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Контуры -> bbox
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area and area > best_area:
            best_area = area
            best = c

    if best is None:
        return mask, None

    x, y, w, h = cv2.boundingRect(best)
    return mask, (x, y, w, h)

def crop_rect(img: np.ndarray, x: int, y: int, cw: int, ch: int):
    h, w = img.shape[:2]
    if cw <= 0 or ch <= 0:
        raise ValueError("Ширина/высота вырезки должны быть > 0.")
    if x < 0 or y < 0 or x >= w or y >= h:
        raise ValueError("Координаты (x, y) должны находиться внутри изображения.")
    x2 = x + cw
    y2 = y + ch
    if x2 > w or y2 > h:
        raise ValueError("Прямоугольник вырезки выходит за пределы изображения.")
    return img[y:y2, x:x2].copy()

def crop_mask(img: np.ndarray, mask: np.ndarray):
    """mask: single-channel uint8 0..255, same size as image"""
    h, w = img.shape[:2]
    if mask is None:
        raise ValueError("Маска не задана.")
    if mask.ndim != 2:
        raise ValueError("Маска должна быть одноканальной (grayscale).")
    if mask.shape[:2] != (h, w):
        raise ValueError("Размер маски должен совпадать с размером изображения.")
    # применим маску: где mask>0 оставляем, иначе делаем черным
    out = img.copy()
    out[mask == 0] = 0
    return out

def flip_image(img: np.ndarray, mode: str):
    if mode == "Horizontal":
        return cv2.flip(img, 1)
    if mode == "Vertical":
        return cv2.flip(img, 0)
    if mode == "Both":
        return cv2.flip(img, -1)
    return img

def rotate_image(img: np.ndarray, angle_deg: float, center_x: int, center_y: int):
    h, w = img.shape[:2]
    cx = clamp(center_x, 0, w - 1)
    cy = clamp(center_y, 0, h - 1)
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    # bilinear by requirement
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def adjust_brightness_contrast(img: np.ndarray, brightness: int, contrast: int):
    """
    brightness: -100..100
    contrast: -100..100
    """
    # alpha (contrast), beta (brightness)
    # typical mapping
    alpha = 1.0 + (contrast / 100.0)
    beta = brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def adjust_color_balance(img: np.ndarray, r: float, g: float, b: float):
    """Multipliers, e.g. 0.0..2.0"""
    out = img.astype(np.float32)
    # BGR order in OpenCV
    out[:, :, 0] *= b
    out[:, :, 1] *= g
    out[:, :, 2] *= r
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def add_gaussian_noise(img: np.ndarray, sigma: float):
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def add_salt_pepper(img: np.ndarray, amount: float, s_vs_p: float):
    """
    amount: fraction of pixels to alter (0..1)
    s_vs_p: salt vs pepper ratio (0..1)
    """
    if amount <= 0:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    num = int(amount * h * w)

    num_salt = int(num * s_vs_p)
    num_pepper = num - num_salt

    # Salt
    ys = np.random.randint(0, h, num_salt)
    xs = np.random.randint(0, w, num_salt)
    out[ys, xs] = 255

    # Pepper
    yp = np.random.randint(0, h, num_pepper)
    xp = np.random.randint(0, w, num_pepper)
    out[yp, xp] = 0

    return out

def blur_image(img: np.ndarray, mode: str, k: int, sigma: float):
    if k <= 0:
        raise ValueError("Размер ядра должен быть > 0.")
    if k % 2 == 0:
        k += 1  # удобнее: делаем нечётным
    if mode == "Mean":
        return cv2.blur(img, (k, k))
    if mode == "Gaussian":
        if sigma <= 0:
            sigma = 0  # OpenCV подберёт
        return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)
    if mode == "Median":
        return cv2.medianBlur(img, k)
    return img

def encode_image(img: np.ndarray, fmt: str, jpeg_quality: int):
    fmt = fmt.lower()
    if fmt in ("jpg", "jpeg"):
        q = clamp(jpeg_quality, 1, 100)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("Не удалось закодировать JPEG.")
        return buf.tobytes(), "image/jpeg", "jpg"
    if fmt == "png":
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError("Не удалось закодировать PNG.")
        return buf.tobytes(), "image/png", "png"
    if fmt in ("tif", "tiff"):
        ok, buf = cv2.imencode(".tiff", img)
        if not ok:
            raise RuntimeError("Не удалось закодировать TIFF.")
        return buf.tobytes(), "image/tiff", "tiff"
    raise ValueError("Неподдерживаемый формат сохранения.")


# -------------------- UI --------------------
st.title("Онлайн графический редактор (OpenCV)")

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("1) Загрузка")
    up = st.file_uploader(
        "Загрузите изображение (JPEG/PNG/TIFF/…)",
        type=list(SUPPORTED_EXT),
        accept_multiple_files=False
    )

    if up is None:
        st.info("Загрузите файл, чтобы начать.")
        st.stop()

    # Validate extension (Streamlit already filters, but keep defensive)
    ext = os.path.splitext(up.name)[1].lower().strip(".")
    if ext not in SUPPORTED_EXT:
        st.error("Неподдерживаемый формат файла.")
        st.stop()

    # Decode
    try:
        raw = up.read()
        pil = Image.open(io.BytesIO(raw))
        pil.load()  # force decode to catch corrupted files
        img0 = pil_to_bgr(pil)
        if img0 is None or img0.size == 0:
            raise ValueError("Пустое изображение.")
    except Exception:
        st.error("Ошибка загрузки: файл повреждён или формат не поддерживается декодером.")
        with st.expander("Технические детали"):
            st.code(traceback.format_exc())
        st.stop()

    st.caption(f"Файл: {up.name} | размер: {img0.shape[1]}×{img0.shape[0]}")

    st.subheader("2) Параметры обработки")

    # --- Resize ---
    with st.expander("Изменение размера", expanded=False):
        resize_enable = st.checkbox("Включить изменение размера", value=False)
        resize_mode = st.radio("Режим", ["Scale", "Exact"], horizontal=True)
        interp_choice = st.selectbox("Интерполяция", ["Auto", "Nearest", "Bilinear", "Bicubic"], index=0)

        scale = 1.0
        out_w = img0.shape[1]
        out_h = img0.shape[0]

        if resize_mode == "Scale":
            scale = st.slider("Коэффициент масштабирования", 0.1, 5.0, 1.0, 0.1)
        else:
            out_w = st.number_input("Ширина (px)", min_value=1, value=int(img0.shape[1]), step=1)
            out_h = st.number_input("Высота (px)", min_value=1, value=int(img0.shape[0]), step=1)

    # --- Crop rectangle ---
    with st.expander("Вырезка прямоугольника", expanded=False):
        crop_rect_enable = st.checkbox("Включить вырезку прямоугольника", value=False)
        x = st.number_input("x (левый верхний)", min_value=0, value=0, step=1)
        y = st.number_input("y (левый верхний)", min_value=0, value=0, step=1)
        cw = st.number_input("width", min_value=1, value=min(200, int(img0.shape[1])), step=1)
        ch = st.number_input("height", min_value=1, value=min(200, int(img0.shape[0])), step=1)

    # --- Crop by mask ---
    with st.expander("Вырезка произвольной формы (маска)", expanded=False):
        crop_mask_enable = st.checkbox("Включить маску", value=False)

        mask_mode = st.radio(
            "Способ задания маски",
            ["Нарисовать в браузере" if HAS_CANVAS else "Нарисовать (нужно установить streamlit-drawable-canvas)",
             "Загрузить PNG-маску"],
            index=0 if HAS_CANVAS else 1
        )

        mask = None
        if crop_mask_enable:
            if HAS_CANVAS and mask_mode == "Нарисовать в браузере":
                st.caption("Рисуйте белым по тем областям, которые нужно оставить. Чёрное будет удалено.")
                canvas_res = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.0)",
                    stroke_width=15,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    background_image=Image.fromarray(bgr_to_rgb(img0)),
                    update_streamlit=True,
                    height=min(500, img0.shape[0]),
                    width=min(700, img0.shape[1]),
                    drawing_mode="freedraw",
                    key="canvas",
                )
                if canvas_res.image_data is not None:
                    # canvas gives RGBA in displayed size; need to map to original size
                    drawn = canvas_res.image_data.astype(np.uint8)
                    gray = cv2.cvtColor(drawn, cv2.COLOR_RGBA2GRAY)
                    # Resize mask back to original size
                    mask = cv2.resize(gray, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # Binarize
                    mask = (mask > 10).astype(np.uint8) * 255
            else:
                mask_up = st.file_uploader("Загрузите маску PNG (белое = оставить, чёрное = убрать)", type=["png"])
                if mask_up is not None:
                    try:
                        mraw = mask_up.read()
                        mpil = Image.open(io.BytesIO(mraw)).convert("L")
                        marr = np.array(mpil)
                        # ensure size
                        mask = cv2.resize(marr, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask = (mask > 127).astype(np.uint8) * 255
                    except Exception:
                        st.error("Не удалось прочитать маску PNG.")
                        with st.expander("Технические детали"):
                            st.code(traceback.format_exc())

    # --- Flip ---
    with st.expander("Зеркальное отражение", expanded=False):
        flip_mode = st.selectbox("Режим отражения", ["None", "Horizontal", "Vertical", "Both"], index=0)

    # --- Rotate ---
    with st.expander("Поворот", expanded=False):
        rotate_enable = st.checkbox("Включить поворот", value=False)
        angle = st.slider("Угол (градусы)", -180.0, 180.0, 0.0, 1.0)
        cx = st.number_input("Центр X", min_value=0, value=int(img0.shape[1] // 2), step=1)
        cy = st.number_input("Центр Y", min_value=0, value=int(img0.shape[0] // 2), step=1)

    # --- Brightness/contrast ---
    with st.expander("Яркость и контраст", expanded=False):
        bc_enable = st.checkbox("Включить яркость/контраст", value=False)
        brightness = st.slider("Яркость", -100, 100, 0, 1)
        contrast = st.slider("Контраст", -100, 100, 0, 1)

    # --- Color balance ---
    with st.expander("Цветовой баланс (RGB)", expanded=False):
        cb_enable = st.checkbox("Включить цветовой баланс", value=False)
        r = st.slider("R множитель", 0.0, 2.0, 1.0, 0.01)
        g = st.slider("G множитель", 0.0, 2.0, 1.0, 0.01)
        b = st.slider("B множитель", 0.0, 2.0, 1.0, 0.01)

    # --- Noise ---
    with st.expander("Добавление шума", expanded=False):
        noise_mode = st.selectbox("Тип шума", ["None", "Gaussian", "Salt & Pepper"], index=0)
        sigma = st.slider("Sigma (Gaussian)", 0.0, 50.0, 10.0, 0.5)
        amount = st.slider("Amount (Salt&Pepper)", 0.0, 0.2, 0.02, 0.005)
        s_vs_p = st.slider("Salt vs Pepper", 0.0, 1.0, 0.5, 0.01)

    # --- Blur ---
    with st.expander("Размытие", expanded=False):
        blur_mode = st.selectbox("Тип размытия", ["None", "Mean", "Gaussian", "Median"], index=0)
        k = st.slider("Размер ядра (k)", 1, 51, 7, 2)
        blur_sigma = st.slider("Sigma (Gaussian blur)", 0.0, 20.0, 0.0, 0.5)

    st.subheader("3) Сохранение")
    out_fmt = st.selectbox("Формат", ["jpg", "png", "tiff"], index=1)
    jpeg_quality = 95
    if out_fmt == "jpg":
        jpeg_quality = st.slider("Качество JPEG", 1, 100, 95, 1)


    # --- Color space conversion ---
    with st.expander("Цветовое пространство", expanded=False):
        cs_enable = st.checkbox("Включить преобразование цветового пространства", value=False)
        cs_target = st.selectbox(
            "Перевести изображение в",
            ["BGR", "RGB", "GRAY", "HSV", "LAB", "YCrCb"],
            index=0
        )
        cs_show_channels = st.checkbox("Показать каналы отдельно (только для превью)", value=False)

    # --- Find object by color ---
    with st.expander("Поиск объекта по цвету", expanded=False):
        find_enable = st.checkbox("Включить поиск по цвету", value=False)

        find_space = st.radio("Цвет задан в", ["RGB", "HSV"], horizontal=True)

        target_rgb = None
        target_hsv = None

        if find_space == "RGB":
            picked = st.color_picker("Цвет объекта (RGB)", value="#ff0000")
            # hex -> (R,G,B)
            pr = int(picked[1:3], 16)
            pg = int(picked[3:5], 16)
            pb = int(picked[5:7], 16)
            target_rgb = (pr, pg, pb)
            tol_rgb = st.slider("Допуск ± (по каналам RGB)", 0, 120, 40, 1)

            # фиктивные, чтобы ниже код был единым
            tol_h = 10
            tol_s = 40
            tol_v = 40

        else:
            st.caption("HSV в OpenCV: H 0..179, S 0..255, V 0..255")
            h0 = st.slider("H", 0, 179, 0, 1)
            s0 = st.slider("S", 0, 255, 255, 1)
            v0 = st.slider("V", 0, 255, 255, 1)
            target_hsv = (h0, s0, v0)

            tol_h = st.slider("Допуск H (±)", 0, 40, 15, 1)
            tol_s = st.slider("Допуск S (±)", 0, 255, 60, 1)
            tol_v = st.slider("Допуск V (±)", 0, 255, 60, 1)

            tol_rgb = 40  # фиктивный

        min_area = st.number_input("Минимальная площадь объекта (px²)", min_value=0, value=80, step=20)

        morph_open_k = st.slider("Морфология: OPEN (очистка шума), k", 0, 31, 0, 2)
        morph_close_k = st.slider("Морфология: CLOSE (заливка дыр), k", 0, 31, 0, 2)

        find_action = st.selectbox(
            "Что сделать с найденным объектом",
            ["Показать рамку и координаты", "Обрезать изображение по объекту"],
            index=0
        )


# -------------------- Processing pipeline --------------------
with right:
    st.subheader("Превью")

    try:
        img = img0.copy()

        # Resize
        if resize_enable:
            img = resize_image(
                img,
                mode=resize_mode,
                scale=float(scale),
                out_w=int(out_w),
                out_h=int(out_h),
                interp_choice=interp_choice
            )

        # Rect crop
        if crop_rect_enable:
            img = crop_rect(img, int(x), int(y), int(cw), int(ch))

        # Mask crop (applies to current image; if you want it always based on original, move earlier)
        if crop_mask_enable:
            # mask should match current img size
            if mask is None:
                raise ValueError("Маска включена, но не задана/не построена.")
            # resize mask if image was resized/cropped
            mh, mw = mask.shape[:2]
            ih, iw = img.shape[:2]
            if (mh, mw) != (ih, iw):
                mask2 = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
            else:
                mask2 = mask
            img = crop_mask(img, mask2)

        # Flip
        if flip_mode != "None":
            img = flip_image(img, flip_mode)

        # Rotate
        if rotate_enable:
            # if image size changed, center inputs may be out of date — clamp inside
            img = rotate_image(img, float(angle), int(cx), int(cy))

        # Brightness/Contrast
        if bc_enable:
            img = adjust_brightness_contrast(img, int(brightness), int(contrast))

        # Color balance
        if cb_enable:
            img = adjust_color_balance(img, float(r), float(g), float(b))

        # Noise
        if noise_mode == "Gaussian":
            img = add_gaussian_noise(img, float(sigma))
        elif noise_mode == "Salt & Pepper":
            img = add_salt_pepper(img, float(amount), float(s_vs_p))

        # Blur
        if blur_mode != "None":
            img = blur_image(img, blur_mode, int(k), float(blur_sigma))

        # --- Find object by color (on current image) ---
        found_bbox = None
        found_mask = None

        if find_enable:
            found_mask, found_bbox = find_object_by_color(
                img_bgr=img,
                input_space=find_space,
                target_rgb=target_rgb,
                target_hsv=target_hsv,
                tol_h=int(tol_h),
                tol_s=int(tol_s),
                tol_v=int(tol_v),
                tol_rgb=int(tol_rgb),
                min_area=int(min_area),
                morph_open_k=int(morph_open_k),
                morph_close_k=int(morph_close_k),
            )

            # Покажем маску (полезно для отладки)
            st.image(found_mask, caption="Маска найденного цвета (white = объект)", use_container_width=True)

            if found_bbox is None:
                st.warning("Объект по указанному цвету не найден (или меньше min_area).")
            else:
                x1, y1, w1, h1 = found_bbox
                x2, y2 = x1 + w1, y1 + h1

                if find_action == "Показать рамку и координаты":
                    img = img.copy()
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img, f"x={x1}, y={y1}, w={w1}, h={h1}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                    )
                    st.write(f"Найденный объект: x={x1}, y={y1}, w={w1}, h={h1} (x2={x2}, y2={y2})")

                elif find_action == "Обрезать изображение по объекту":
                    img = img[y1:y2, x1:x2].copy()
                    st.write(f"Обрезка по bbox: ({x1}, {y1}) .. ({x2}, {y2})")

        # --- Color space conversion (final) ---
        img_for_save = img
        preview_data = bgr_to_rgb(img)
        preview_caption = f"Результат: {img.shape[1]}×{img.shape[0]}"

        if cs_enable:
            converted = convert_from_bgr(img, cs_target)

            # Показываем данные выбранного пространства.
            # Для HSV/LAB/YCrCb это псевдоцветовое представление (как есть), для RGB/GRAY — привычный вид.
            if cs_target.upper() == "BGR":
                preview_data = bgr_to_rgb(converted)
                preview_caption = f"BGR (превью): {img.shape[1]}×{img.shape[0]}"
            elif cs_target.upper() == "RGB":
                preview_data = converted
                preview_caption = f"RGB: {img.shape[1]}×{img.shape[0]}"
            elif cs_target.upper() == "GRAY":
                preview_data = converted
                preview_caption = f"GRAY: {img.shape[1]}×{img.shape[0]}"
            else:
                preview_data = converted
                preview_caption = f"{cs_target}: псевдоцветовое превью ({img.shape[1]}×{img.shape[0]})"

            # Для сохранения кодируем стандартное изображение в BGR.
            img_for_save = convert_to_bgr(converted, cs_target)

            # Каналы отдельно (только превью)
            if cs_show_channels:
                if converted.ndim == 2:
                    st.image(converted, caption=f"{cs_target}: канал (GRAY)", use_container_width=True)
                else:
                    ch_names = {
                        "BGR": ["B", "G", "R"],
                        "RGB": ["R", "G", "B"],
                        "HSV": ["H", "S", "V"],
                        "LAB": ["L", "a", "b"],
                        "YCRCB": ["Y", "Cr", "Cb"],
                    }.get(cs_target.upper(), ["C1", "C2", "C3"])
                    for i, nm in enumerate(ch_names):
                        st.image(converted[:, :, i], caption=f"{cs_target}: канал {nm}", use_container_width=True)

        # Show preview
        st.image(preview_data, caption=preview_caption, use_container_width=True)
        # Save / Download
        try:
            data, mime, ext2 = encode_image(img_for_save, out_fmt, int(jpeg_quality))
            st.download_button(
                label=f"Скачать результат ({out_fmt.upper()})",
                data=data,
                file_name=f"result.{ext2}",
                mime=mime
            )
        except Exception as e:
            st.error(f"Ошибка сохранения/кодирования: {e}")

    except Exception as e:
        st.error(f"Ошибка обработки: {e}")
        with st.expander("Технические детали"):
            st.code(traceback.format_exc())