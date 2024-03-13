import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Scale, Label

class ArtifactExtractionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Artifact Extraction App")

        self.low_cutoff = 50
        self.high_cutoff = 100
        self.threshold_value = 128
        self.image = None

        self.create_widgets()

    def create_widgets(self):
        self.load_button = tk.Button(self.master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.low_cutoff_label = Label(self.master, text="Low Cutoff")
        self.low_cutoff_label.pack()
        self.low_cutoff_scale = Scale(self.master, from_=0, to=255, orient="horizontal", length=300)
        self.low_cutoff_scale.set(self.low_cutoff)
        self.low_cutoff_scale.pack()

        self.high_cutoff_label = Label(self.master, text="High Cutoff")
        self.high_cutoff_label.pack()
        self.high_cutoff_scale = Scale(self.master, from_=0, to=255, orient="horizontal", length=300)
        self.high_cutoff_scale.set(self.high_cutoff)
        self.high_cutoff_scale.pack()

        self.threshold_label = Label(self.master, text="Threshold")
        self.threshold_label.pack()
        self.threshold_scale = Scale(self.master, from_=0, to=255, orient="horizontal", length=300)
        self.threshold_scale.set(self.threshold_value)
        self.threshold_scale.pack()

        self.apply_button = tk.Button(self.master, text="Apply", command=self.apply_filters)
        self.apply_button.pack()

        self.save_button = tk.Button(self.master, text="Save", command=self.save_image)
        self.save_button.pack()

        self.image_label = tk.Label(self.master)
        self.image_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            cv2.imshow("Original Image", self.image)

    def apply_filters(self):
        if self.image is not None:
            self.low_cutoff = self.low_cutoff_scale.get()
            self.high_cutoff = self.high_cutoff_scale.get()
            self.threshold_value = self.threshold_scale.get()

            bandpass_image = self.bandpass_filter(self.image, self.low_cutoff, self.high_cutoff)
            binary_image = self.binarize_image(bandpass_image, self.threshold_value)

            cv2.imshow("Processed Image", bandpass_image)
            cv2.imshow("Binary Image", binary_image)

        else:
            messagebox.showerror("Error", "Please load an image first.")

    def bandpass_filter(self, image, low_cutoff, high_cutoff):
        fft_image = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft_image)

        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2  # 中心のピクセルのインデックス

        # 中心のピクセルのインデックスを整数に変換
        center_row = int(center_row)
        center_col = int(center_col)

        lp_mask = np.zeros((rows, cols), np.uint8)
        lp_mask[center_row - high_cutoff:center_row + high_cutoff, center_col - high_cutoff:center_col + high_cutoff] = 1

        hp_mask = np.ones((rows, cols), np.uint8) - lp_mask

        filtered_fft_shifted = fft_shifted * hp_mask

        filtered_image_fft = np.fft.ifftshift(filtered_fft_shifted)
        filtered_image = np.fft.ifft2(filtered_image_fft)
        filtered_image = np.abs(filtered_image)

        return filtered_image

    def binarize_image(self, image, threshold_value):
        _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        return binary_image

    def save_image(self):
        if self.image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
            if file_path:
                binary_image = self.binarize_image(self.image, self.threshold_value)
                cv2.imwrite(file_path, binary_image)
                messagebox.showinfo("Success", "Image saved successfully.")
        else:
            messagebox.showerror("Error", "No image to save.")

def main():
    root = tk.Tk()
    app = ArtifactExtractionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
