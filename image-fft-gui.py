import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
from scipy.fft import fft2, ifft2
import numpy as np

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        
        # メインフレーム
        main_frame = ttk.Frame(root)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # 操作フレーム
        control_frame = ttk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # ボタンフレーム
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        # 変換ボタン
        self.convert_button = ttk.Button(button_frame, text="変換", command=self.apply_filter)
        self.convert_button.pack(fill=tk.X)
        
        # フィルタ設定フレーム
        filter_frame = ttk.LabelFrame(control_frame, text="フィルタ設定")
        filter_frame.pack(pady=10, fill=tk.Y)
        
        # 縦方向フィルタ設定
        self.vertical_filter_frame = ttk.Frame(filter_frame)
        self.vertical_filter_frame.pack(pady=5)
        
        ttk.Label(self.vertical_filter_frame, text="縦方向バンドパスフィルタ").grid(row=0, column=0, columnspan=2)
        ttk.Label(self.vertical_filter_frame, text="上限").grid(row=1, column=0, sticky=tk.E)
        self.vertical_upper_slider = ttk.Scale(self.vertical_filter_frame, from_=0, to=2000, length=150)
        self.vertical_upper_slider.grid(row=1, column=1)
        ttk.Label(self.vertical_filter_frame, text="下限").grid(row=2, column=0, sticky=tk.E)
        self.vertical_lower_slider = ttk.Scale(self.vertical_filter_frame, from_=0, to=2000, length=150)
        self.vertical_lower_slider.grid(row=2, column=1)
        # 縦方向上限スライダーの値を表示するラベル
        self.vertical_upper_value_label = ttk.Label(self.vertical_filter_frame, textvariable=tk.StringVar(value="0"))
        self.vertical_upper_value_label.grid(row=1, column=2)
        # 縦方向下限スライダーの値を表示するラベル
        self.vertical_lower_value_label = ttk.Label(self.vertical_filter_frame, textvariable=tk.StringVar(value="0"))
        self.vertical_lower_value_label.grid(row=2, column=2)
        
        # 横方向フィルタ設定
        self.horizontal_filter_frame = ttk.Frame(filter_frame)
        self.horizontal_filter_frame.pack(pady=5)
        
        ttk.Label(self.horizontal_filter_frame, text="横方向バンドパスフィルタ").grid(row=0, column=0, columnspan=2)
        ttk.Label(self.horizontal_filter_frame, text="上限").grid(row=1, column=0, sticky=tk.E)
        self.horizontal_upper_slider = ttk.Scale(self.horizontal_filter_frame, from_=0, to=2000, length=150)
        self.horizontal_upper_slider.grid(row=1, column=1)
        ttk.Label(self.horizontal_filter_frame, text="下限").grid(row=2, column=0, sticky=tk.E)
        self.horizontal_lower_slider = ttk.Scale(self.horizontal_filter_frame, from_=0, to=2000, length=150)
        self.horizontal_lower_slider.grid(row=2, column=1)
        # 横方向上限スライダーの値を表示するラベル
        self.horizontal_upper_value_label = ttk.Label(self.horizontal_filter_frame, textvariable=tk.StringVar(value="0"))
        self.horizontal_upper_value_label.grid(row=1, column=2)
        # 横方向下限スライダーの値を表示するラベル
        self.horizontal_lower_value_label = ttk.Label(self.horizontal_filter_frame, textvariable=tk.StringVar(value="0"))
        self.horizontal_lower_value_label.grid(row=2, column=2)
        
        # 二値化設定
        ttk.Label(filter_frame, text="二値化設定").pack(pady=5)
        ttk.Label(filter_frame, text="閾値").pack()
        self.threshold_slider = ttk.Scale(filter_frame, from_=0, to=255, length=150)
        self.threshold_slider.pack()
        # 二値化の閾値スライダーの値を表示するラベル
        self.threshold_value_label = ttk.Label(filter_frame, textvariable=tk.StringVar(value="0"))
        self.threshold_value_label.pack()
        
        # 画像表示フレーム
        self.image_frame = ttk.LabelFrame(main_frame, text="画像表示")
        self.image_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # タブコンテナ
        self.tab_control = ttk.Notebook(self.image_frame)
        self.tab_control.pack(expand=True, fill=tk.BOTH)
        
        # フーリエ変換タブ
        self.fft_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.fft_tab, text="フーリエ変換")
        self.fft_image_label = ttk.Label(self.fft_tab)
        self.fft_image_label.pack(expand=True, fill=tk.BOTH)
        
        # 逆フーリエ変換タブ
        self.ifft_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.ifft_tab, text="逆フーリエ変換")
        self.ifft_image_label = ttk.Label(self.ifft_tab)
        self.ifft_image_label.pack(expand=True, fill=tk.BOTH)
        
        # 二値化タブ
        self.binary_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.binary_tab, text="二値化")
        self.binary_image_label = ttk.Label(self.binary_tab)
        self.binary_image_label.pack(expand=True, fill=tk.BOTH)
        
        # 画像の読み込みと初期表示
        self.image = Image.open("img_origin.jpg")
        self.display_image(self.image, self.fft_image_label)
        self.display_image(self.image, self.ifft_image_label)
        self.display_image(self.image, self.binary_image_label)
        
        # スライダーの値が変更されたときに更新関数を呼び出す
        self.vertical_upper_slider.config(command=self.update_slider_value_labels)
        self.vertical_lower_slider.config(command=self.update_slider_value_labels)
        self.horizontal_upper_slider.config(command=self.update_slider_value_labels)
        self.horizontal_lower_slider.config(command=self.update_slider_value_labels)
        self.threshold_slider.config(command=self.update_slider_value_labels)
    
    # 各スライダーの値が変更されたときに更新する関数
    def update_slider_value_labels(self,*args):
        self.vertical_upper_value_label["text"] = str(self.vertical_upper_slider.get())
        self.vertical_lower_value_label["text"] = str(self.vertical_lower_slider.get())
        self.horizontal_upper_value_label["text"] = str(self.horizontal_upper_slider.get())
        self.horizontal_lower_value_label["text"] = str(self.horizontal_lower_slider.get())
        self.threshold_value_label["text"] = str(self.threshold_slider.get())
        print("Vertical upper:", self.vertical_upper_slider.get())
        print("Vertical lower:", self.vertical_lower_slider.get())
        print("Horizontal upper:", self.horizontal_upper_slider.get())
        print("Horizontal lower:", self.horizontal_lower_slider.get())
        print("Threshold:", self.threshold_slider.get())
    def apply_filter(self):
        # フィルタ設定の取得
        vertical_upper = self.vertical_upper_slider.get()
        vertical_lower = self.vertical_lower_slider.get()
        horizontal_upper = self.horizontal_upper_slider.get()
        horizontal_lower = self.horizontal_lower_slider.get()
        threshold = self.threshold_slider.get()
        
        # 画像をnumpy配列に変換
        img_array = np.array(self.image)
        
        # 画像をグレースケールに変換
        gray_img_array = img_array.mean(axis=2)
        
        # フーリエ変換
        fft_img = fft2(gray_img_array)
        
        # バンドパスフィルタ適用
        fft_img_filtered = np.zeros_like(fft_img)
        fft_img_filtered[int(vertical_lower):int(vertical_upper), int(horizontal_lower):int(horizontal_upper)] = fft_img[int(vertical_lower):int(vertical_upper), int(horizontal_lower):int(horizontal_upper)]

        
        # 逆フーリエ変換
        ifft_img = ifft2(fft_img_filtered).real
        
        # 二値化
        binary_img = np.where(ifft_img < threshold, 0, 255).astype(np.uint8)
        
        # 画像をPIL形式に戻す
        fft_img_pil = Image.fromarray(np.abs(fft_img_filtered))
        ifft_img_pil = Image.fromarray(binary_img)
        binary_img_pil = Image.fromarray(binary_img)
        
        # 画像を表示
        self.display_image(fft_img_pil, self.fft_image_label)
        self.display_image(ifft_img_pil, self.ifft_image_label)
        self.display_image(binary_img_pil, self.binary_image_label)
        
    def display_image(self, image, label):
        # 画像をリサイズしてLabelに表示
        image.thumbnail((600, 600))
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo
        
def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
