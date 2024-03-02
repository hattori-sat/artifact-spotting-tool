import cv2
import numpy as np

def extract_artifacts(image_path):
    # 画像の読み込み
    image = cv2.imread(image_path)

    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SIFT特徴量抽出器を作成
    sift = cv2.SIFT_create()

    # 画像からSIFT特徴量を抽出
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 特徴量を描画
    output_image = cv2.drawKeypoints(gray, keypoints, None)

    return output_image

# 入力画像のパス
input_image_path = 'img_origin.JPG'

# アーティファクトの特徴を抽出した結果を取得
result_image = extract_artifacts(input_image_path)

# 結果を保存
output_image_path = 'output_artifacts_image.jpg'
cv2.imwrite(output_image_path, result_image)
