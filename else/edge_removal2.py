import cv2
import numpy as np

def remove_edges(image_path):
    # 画像の読み込み
    image = cv2.imread(image_path)

    # ガウシアンフィルタを適用してノイズを除去
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # グレースケールに変換
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Cannyエッジ検出アルゴリズムを適用して、画像内のエッジを検出
    edges = cv2.Canny(gray, 50, 80)

    # 二値化して完全な二値画像を得る
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)

    return binary

# 入力画像のパス
input_image_path = 'img_origin.JPG'

# エッジを除去した結果を取得
result_image = remove_edges(input_image_path)

# 結果を保存
output_image_path = 'output_binary_image_50_80.jpg'
cv2.imwrite(output_image_path, result_image)
