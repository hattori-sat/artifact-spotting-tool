import cv2
import numpy as np

def remove_edges(image_path):
    # 画像の読み込み
    image = cv2.imread(image_path)

    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # エッジ検出アルゴリズムを適用して、縦線（ボーダー）を検出する
    edges = cv2.Canny(gray, 100, 120)

    # エッジを除去する
    edges_removed = cv2.bitwise_not(edges)

    # 元の画像からエッジを除去する
    result = cv2.bitwise_and(image, image, mask=edges_removed)

    return result

# 入力画像のパス
input_image_path = 'img_origin.jpg'

# エッジを除去した結果を取得
result_image = remove_edges(input_image_path)

# 結果を保存
output_image_path = 'output_image_100_120.jpg'
cv2.imwrite(output_image_path, result_image)
