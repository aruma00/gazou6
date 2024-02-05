import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread('時計1/時計1.jpg')

# グレースケール変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ぼかしを適用してノイズを減少させる
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 二値化
_, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

# 輪郭検出
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 時計の中心を仮定
clock_center = (image.shape[1] // 2, image.shape[0] // 2)

# 12時と6時を結ぶ軸線を設定
clock_axis = (clock_center[0], clock_center[1] + 100)  # 例として、中心から下に100ピクセル移動

# 輪郭の面積の閾値
threshold_area = 1000  # 調整

# 各輪郭から時針と分針の位置を抽出して角度を計算
for contour in contours:
    # 輪郭の面積が一定以下の場合は無視
    if cv2.contourArea(contour) < threshold_area:
        continue

    # 輪郭の中心座標を取得
    M = cv2.moments(contour)
    if M["m00"] == 0:
        continue
    hand_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # 時針と軸線の間の角度を計算
    angle_hour = np.degrees(np.arctan2(clock_axis[1] - hand_center[1], hand_center[0] - clock_axis[0]))
    # 分針と軸線の間の角度を計算
    angle_minute = np.degrees(np.arctan2(hand_center[1] - clock_axis[1], hand_center[0] - clock_axis[0]))
    
    # 角度から時間および分を計算
    hour = int((angle_hour + 360) % 360 / 30)  # 1時間あたり30度
    minute = int((angle_minute + 360) % 360 / 6)  # 1分あたり6度

    print(f'時計1の時刻: {hour}時{minute}分')

    
# 結果を表示
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 画像の読み込み
image = cv2.imread('時計1/時計2.jpg')

# グレースケール変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ぼかしを適用してノイズを減少させる
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 二値化
_, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

# 輪郭検出
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 時計の中心を見つける（仮定）
clock_center = (image.shape[1] // 2, image.shape[0] // 2)

# 12時と6時を結ぶ軸線を設定
clock_axis = (clock_center[0], clock_center[1] + 100)  # 例として、中心から下に100ピクセル移動

# 検出された輪郭を描画
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# 最も大きな輪郭を選択
largest_contour = max(contours, key=cv2.contourArea)

# 輪郭から時針と分針の位置を抽出して角度を計算
hand_center = tuple(largest_contour[0][0])  # 仮の例として輪郭の最初の点を取得

# 時針と軸線の間の角度を計算
angle_hour = np.degrees(np.arctan2(hand_center[1] - clock_axis[1], hand_center[0] - clock_axis[0]))

# 分針と軸線の間の角度を計算
angle_minute = np.degrees(np.arctan2(clock_axis[1] - hand_center[1], hand_center[0] - clock_axis[0]))

# 角度から時間および分を計算
hour = int((angle_hour + 360) % 360 / 30)  # 1時間あたり30度
minute = int((angle_minute + 360) % 360 / 6)  # 1分あたり6度

print(f'時計2の時刻: {hour}時{minute}分')

# 結果を表示
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 画像の読み込み
image = cv2.imread('時計1/時計3.jpg')

# グレースケール変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ぼかしを適用してノイズを減少させる
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# エッジ検出（Canny）
edges = cv2.Canny(blurred, 50, 150)

# 輪郭検出
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 12時と6時を結ぶ軸線を設定
clock_axis = (image.shape[1] // 2 , image.shape[0] // 2 +100)  # 例として、中心から下に100ピクセル移動

# 輪郭の面積の閾値
threshold_area = 1000  # 調整

# 大きな輪郭を格納する変数
largest_contour = None
largest_contour_area = 0

# 各輪郭から時針と分針の位置を抽出して角度を計算
for contour in contours:
    # 輪郭の面積が一定以下の場合は無視
    if cv2.contourArea(contour) < threshold_area:
        continue

    # 輪郭の中心座標を取得
    M = cv2.moments(contour)
    if M["m00"] == 0:
        continue
    hand_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # 輪郭の面積が最大の場合、更新
    if cv2.contourArea(contour) > largest_contour_area:
        largest_contour = contour
        largest_contour_area = cv2.contourArea(contour)

# 大きな輪郭が見つかった場合
if largest_contour is not None:
    # 中心を正確に計算
    ellipse = cv2.fitEllipse(largest_contour)
    hand_center = (int(ellipse[0][0]), int(ellipse[0][1]))

    # 時針と軸線の間の角度を計算
    angle_hour = np.degrees(np.arctan2(clock_axis[1] - hand_center[1], hand_center[0] - clock_axis[0]))
    
    # 分針と軸線の間の角度を計算
    angle_minute = np.degrees(np.arctan2(hand_center[1] - clock_axis[1], hand_center[0] - clock_axis[0]))
    # 角度から時間および分を計算
    hour = int((angle_hour + 360) % 360 / 30)  # 1時間あたり30度
    minute = int((angle_minute + 360) % 360 / 6)  # 1分あたり6度

    print(f'時計3の時刻: {hour}時{minute}分')
    
    # 結果を表示
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 画像の読み込み
image = cv2.imread('時計1/時計4.jpg')

# グレースケール変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ぼかしを適用してノイズを減少させる
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 二値化
_, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

# 輪郭検出
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 時計の中心を見つける（仮定）
clock_center = (image.shape[1] // 2, image.shape[0] // 2)

# 12時と6時を結ぶ軸線を設定
clock_axis = (clock_center[0], clock_center[1] + 100)  # 例として、中心から下に100ピクセル移動

# 検出された輪郭を描画する
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# 最も大きな輪郭を選択
largest_contour = max(contours, key=cv2.contourArea)

# 輪郭から時針と分針の位置を抽出して角度を計算
hand_center = tuple(largest_contour[0][0])  

# 時針と軸線の間の角度を計算
angle_hour = np.degrees(np.arctan2(hand_center[1] - clock_axis[1], hand_center[0] - clock_axis[0]))

# 分針と軸線の間の角度を計算
angle_minute = np.degrees(np.arctan2(clock_axis[1] - hand_center[1], hand_center[0] - clock_axis[0]))

# 角度から時間および分を計算
hour = int((angle_hour + 360) % 360 / 30)  # 1時間あたり30度
minute = int((angle_minute + 360) % 360 / 6)  # 1分あたり6度

print(f'時計4の時刻: {hour}時{minute}分')

# 結果を表示
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
