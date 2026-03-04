import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import numpy as np
import cv2

def open_image():
    global img, img_tk, root
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.tiff;*.bmp")])
    if not file_path:
        return
    try:
        img = Image.open(file_path)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        root.geometry(f"{img.width + 100}x{img.height + 150}")  # Adjust window size
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")

def resize_image():
    global img, img_tk, root
    try:
        width = int(width_entry.get())
        height = int(height_entry.get())
        method = resize_method_var.get()
        interpolation = {
            "Ближайший сосед": Image.NEAREST,
            "Билинейная": Image.BILINEAR,
            "Бикубическая": Image.BICUBIC
        }[method]
        img = img.resize((width, height), interpolation)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        root.geometry(f"{width + 100}x{height + 150}")  # Adjust window size
    except ValueError:
        messagebox.showerror("Ошибка", "Введите корректные размеры!")

def rotate_image():
    global img, img_tk, root
    try:
        angle = int(angle_entry.get())
        img = img.rotate(angle, resample=Image.BILINEAR, center=(img.width//2, img.height//2))
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        root.geometry(f"{img.width + 100}x{img.height + 150}")  # Adjust window size
    except ValueError:
        messagebox.showerror("Ошибка", "Введите корректный угол!")

def flip_image(direction):
    global img, img_tk, root
    if direction == "Горизонтально":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == "Вертикально":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    root.geometry(f"{img.width + 100}x{img.height + 150}")  # Adjust window size

def save_image():
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tiff")])
    if file_path:
        try:
            img.save(file_path)
            messagebox.showinfo("Сохранение", "Изображение сохранено успешно!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении: {e}")

def adjust_brightness(value):
    global img, img_tk
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(float(value))
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

def apply_blur():
    global img, img_tk
    img = img.filter(ImageFilter.GaussianBlur(5))
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

root = tk.Tk()
root.title("Редактор изображений")

frame = tk.Frame(root)
frame.pack(side=tk.LEFT, padx=10, pady=10)

canvas = tk.Canvas(frame)
canvas.pack()

btn_open = tk.Button(root, text="Открыть изображение", command=open_image)
btn_open.pack()

width_entry = tk.Entry(root)
height_entry = tk.Entry(root)
btn_resize = tk.Button(root, text="Изменить размер", command=resize_image)
width_entry.pack()
height_entry.pack()
btn_resize.pack()

angle_entry = tk.Entry(root)
btn_rotate = tk.Button(root, text="Повернуть", command=rotate_image)
angle_entry.pack()
btn_rotate.pack()

btn_flip_h = tk.Button(root, text="Зеркало по горизонтали", command=lambda: flip_image("Горизонтально"))
btn_flip_v = tk.Button(root, text="Зеркало по вертикали", command=lambda: flip_image("Вертикально"))
btn_flip_h.pack()
btn_flip_v.pack()

btn_save = tk.Button(root, text="Сохранить изображение", command=save_image)
btn_save.pack()

brightness_slider = tk.Scale(root, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, label="Яркость", command=adjust_brightness)
brightness_slider.pack()

btn_blur = tk.Button(root, text="Размытие", command=apply_blur)
btn_blur.pack()

resize_method_var = tk.StringVar(value="Билинейная")
resize_method_menu = tk.OptionMenu(root, resize_method_var, "Ближайший сосед", "Билинейная", "Бикубическая")
resize_method_menu.pack()

root.mainloop()