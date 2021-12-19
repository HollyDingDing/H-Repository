# Laser Recognition
## <Laser_recognition模型製作與影像前置處理>
#### Author: Hdingzp4  Tylerj86
* **影像資料抓取**：
  我們的影像檔案都優先存儲於雲端硬碟中的1sec_video資料夾中，由於是使用colab進行編寫，我們引入google.colab.drive將colab掛載至雲端硬碟上以取得data並利於建立database。
  * Run In Colab
    ```python
    # Colab 掛載 google drive /content/gdrive 目錄
    from google.colab import drive
    drive.mount('/content/gdrive')
    ```
    ```python
    # 導入 PyDrive 和相關程式庫
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    import os

    # 驗證並創建 PyDrive 客戶端
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    ```
    ```python
    # 取得資料目錄
    path = '/content/gdrive/MyDrive/laserRecognition/字母辨識/1sec_video/'
    resources_path = f'{path}/Resources/'
    videos = os.listdir(path)
    print(videos)
    ```
  * Run In Jupyter
    ```python
    import os
    ```
    ```python
    # 取得資料目錄
    path = '/1sec_video/'
    resources_path = f'{path}/Resources/'
    videos = os.listdir(path)
    print(videos)
    ```

```python
# 取得所有資料類別
classes_num = 0
classList = []
for item in videos:
  if len(item) == 1:
    classes_num += 1
    classList.append(item)
print(range(classes_num))
```
為符合CNN LSTM模型所需訓練的格式，首先我們利用opencv-python模組進行影片的前置處理。
我們建立了名為Video_process_tool的Class以利處理影像，於其中建立了get_mask resize_img及gray_img三種函式。
get_mask:
針對拍攝的影像設定一個固定的mask以清晰抓取的震動並切割。
resize_img:
將圖片縮放為63x75的影像。
gray_img:
利用opencv的cv2.cvtColor模組先將圖片轉成灰階。
建立frames_extraction函式，使用cv2抓取檔案中的影像，檢測影片抓取的幀數，加以切割並分配長度設定為15幀的影像。再將影像經前述的Video_proccess_tool處理後，對於每個pixel除以255，也就是使其成為介於0到1之間的數值，以方便後續進行卷積或是運算，最後將每幀圖片加入陣列中回傳。
```python
# 影像處理工具
class Video_process_tool:
  def __init__(self):
    pass
  def get_mask(self, img):
      mask = img[img.shape[0] // 2 - 175: img.shape[0] // 2 + 125,
              img.shape[1] // 2 - 125: img.shape[1] // 2 + 125]
      return mask

  def gray_img(self, img):
      return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

vid = Video_process_tool()
```
建立create_database函式，走訪資料夾中的所有影像，調用frames_extraction並取得其回傳的影祥資料加入到同樣標籤的陣列中，分成相對應的features和Labels陣列分別代表該影片的data和其對應的標籤。
使用sklearn對處理好的features和Labels，進行拆分，分成features_train(用於訓練的資料), features_test(用於測試的資料), labels_train(用於訓練的標籤), labels_test(用於測試的標籤), video_files_train(訓練的影片檔案位置), video_files_test(測式的影片檔案位置)。
利用json將檔案dump至指定的json檔案位置，也就是將要使用的database，接著就可進入模型訓練的階段，只需再將json檔案載入就行了。
