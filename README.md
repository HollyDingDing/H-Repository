# Laser Recognition
## <Laser_recognition模型製作與影像前置處理>
#### Author: Hdingzp4  Tylerj86
<tr>

* **All Steps**:<br>
  <a href="#grab_frames">Collect Frame Datas</a><br>
  <a href="#model_establishment">Model Configuration</a><br>

<p backgound-color="gray">Model Configuration</p>

* <span id="grab_frames">**影像資料抓取**</span>：<br>
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
  ```python
  # Discard the output of this cell.
  # This command uses in colab, cannot be used in jupyter
  %%capture

  # Install the required libraries.
  !pip install pafy youtube-dl moviepy
  ```
  ```python
  # 導入所需影像處理和AI模型所需模組
  import cv2
  import pafy
  import math
  import random
  import json
  import numpy as np
  import datetime as dt
  import tensorflow as tf
  from collections import deque
  import matplotlib.pyplot as plt
  import matplotlib.pylab as lab

  from moviepy.editor import *
  %matplotlib inline

  from sklearn.model_selection import train_test_split

  from tensorflow.keras.layers import *
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.utils import to_categorical
  from tensorflow.keras.callbacks import EarlyStopping
  from tensorflow.keras.utils import plot_model
  ```
  ```python
  seed_constant = 25
  np.random.seed(seed_constant)
  random.seed(seed_constant)
  tf.random.set_seed(seed_constant)
  ```
  為符合CNN LSTM模型所需訓練的格式，首先我們利用opencv-python模組進行影片的前置處理。
  我們建立了名為Video_process_tool的Class以利處理影像，於其中建立了get_mask resize_img及gray_img三種函式。
  * get_mask:<br>
    針對拍攝的影像設定一個固定的mask以清晰抓取的震動並切割。
  * resize_img:<br>
    將圖片縮放為63x75的影像。
  * gray_img:<br>
    利用opencv的cv2.cvtColor模組先將圖片轉成灰階。
    建立frames_extraction函式，使用cv2抓取檔案中的影像，檢測影片抓取的幀數，加以切割並分配長度設定為15幀的影像。再將影像經前述的Video_proccess_tool處理後，對於每個pixel除以255，也就是使其成為介於0到1之間的數值，以方便後續進行卷積或是運算，最後將每幀圖片加入陣列中回。
  ```python
  # 配置模型的儲存位置、已有模組檔案
  cnnlstm_name = 'cnnlstm'
  cnnlstm_path = f'{resources_path}/models/{cnnlstm_name}'
  cnnlstm_file = os.listdir(cnnlstm_path)

  lrcn_name = 'lrcn'
  lrcn_path = f'{resources_path}/models/{lrcn_name}'
  lrcn_file = os.listdir(lrcn_path)
  ```
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
  ```python
  # 配置固定的影像大小
  IMAGE_HEIGHT , IMAGE_WIDTH = 63, 75

  # 配置LSTM要處理多長的序列和從影片所需取得的圖片數
  SEQUENCE_LENGTH = 15

  # 配置Database的位置
  DATASET_DIR = path

  # 配置所有Class類別
  CLASSES_LIST = classList
  del classList
  ```
  ```python
  # 影像抓取函式
  def frames_extraction(video_path):

      # Declare a list to store video frames.
      frames_list = []

      # Read the Video File using the VideoCapture object.
      video_reader = cv2.VideoCapture(video_path)

      # Get the total number of frames in the video.
      video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

      # Calculate the the interval after which frames will be added to the list.
      skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

      # Iterate through the Video Frames.
      for frame_counter in range(SEQUENCE_LENGTH):

          # Set the current frame position of the video.
          video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

          # Reading the frame from the video.
          success, frame = video_reader.read()


          # Check if Video frame is not successfully read then break the loop
          if not success:
              break

          frame = cv2.resize(frame,None,fx=0.3,fy=0.3)

          frame = vid.get_mask(frame)
          # print(frame.shape)
          # frame = vid.gray_img(frame)

          # Resize the Frame to fixed height and width.
          resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
          # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
          normalized_frame = resized_frame / 255

          # Append the normalized frame into the frames list
          frames_list.append(normalized_frame)

      # Release the VideoCapture object.
      video_reader.release()

      # Return the frames list.
      return frames_list
  ```
  建立create_database函式，走訪資料夾中的所有影像，調用frames_extraction並取得其回傳的影祥資料加入到同樣標籤的陣列中，分成相對應的features和Labels陣列分別代表該影片的data和其對應的標籤。
  ```python
  # 創建訓練資料集函式
  def create_dataset():
      '''
      This function will extract the data of the selected classes and create the required dataset.
      Returns:
          features:          A list containing the extracted frames of the videos.
          labels:            A list containing the indexes of the classes associated with the videos.
          video_files_paths: A list containing the paths of the videos in the disk.
      '''

      # Declared Empty Lists to store the features, labels and video file path values.
      features = []
      labels = []
      video_files_paths = []

      # Iterating through all the classes mentioned in the classes list
      for class_index, class_name in enumerate(CLASSES_LIST):

          # Get the list of video files present in the specific class name directory.
          files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

          # Display the name of the class whose data is being extracted.
          print(f'Extracting Data of Class: {class_name}, Total File Num: {len(files_list)}')

          # Iterate through all the files present in the files list.
          for num, file_name in enumerate(files_list):

              print(f'Processing {num+1} Data')
              # Get the complete video path.
              video_file_path = os.path.join(DATASET_DIR, class_name, file_name)

              # Extract the frames of the video file.
              frames = frames_extraction(video_file_path)

              # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
              # So ignore the vides having frames less than the SEQUENCE_LENGTH.
              if len(frames) == SEQUENCE_LENGTH:

                  # Append the data to their repective lists.
                  features.append(frames)
                  labels.append(class_index)
                  video_files_paths.append(video_file_path)

      # Converting the list to numpy arrays
      features = np.asarray(features)
      labels = np.array(labels)  

      # Return the frames, class index, and video file path.
      return features, labels, video_files_paths
  ```
  ```python
  # 調用創建資料集功能
  features, labels, video_files_paths = create_dataset()
  ```
  ```python
  # 用 Keras 的 to_categorical 方法把所有類別標籤轉為 one-hot-encoded 向量
  one_hot_encoded_labels = to_categorical(labels)
  print(one_hot_encoded_labels)
  ```
  使用sklearn對處理好的features和Labels，進行拆分，分成features_train(用於訓練的資料), features_test(用於測試的資料), labels_train(用於訓練的標籤), labels_test(用於測試的標籤), video_files_train(訓練的影片檔案位置), video_files_test(測式的影片檔案位置)。
  利用json將檔案dump至指定的json檔案位置，也就是將要使用的database，接著就可進入模型訓練的階段，只需再將json檔案載入就行了。
  ```python
  # 將資料集拆分為訓練和測試資料集
  features_train, features_test, labels_train, labels_test, video_files_train, video_files_test = train_test_split(features, one_hot_encoded_labels, video_files_paths, random_state = seed_constant, train_size=0.8)
  ```
  ```python
  # 把訓練、測試資料用JSON格式寫入檔案中
  with open(f'{resources_path}/features_train.json', 'w') as f:
    json.dump(features_train.tolist(), f)
  with open(f'{resources_path}/features_test.json', 'w') as f:
    json.dump(features_test.tolist(), f)
  with open(f'{resources_path}/labels_train.json', 'w') as f:
    json.dump(labels_train.tolist(), f)
  with open(f'{resources_path}/labels_test.json', 'w') as f:
    json.dump(labels_test.tolist(), f)
  with open(f'{resources_path}/labels.json', 'w') as f:
    json.dump(labels.tolist(), f)
  with open(f'{resources_path}/video_files_train.json', 'w') as f:
    json.dump(video_files_train, f)
  with open(f'{resources_path}/video_files_test.json', 'w') as f:
    json.dump(video_files_test, f)
  ```
* <span id="model_establishment">**模型建立**</span>:<br>
  我們使用CNN LSTM模型來作為預測模型，我們提供兩種方法創建模型，使用兩種不同方法建立模型，來比較不同方法創建的模型好壞，ConvLSTM2D方法創建的模型是CNN、LSTM一起建立，且需要回傳序列(return_sequences=True)，多了時間序列變成4維，需要用MaxPooling3D；LRCN方法創建模型用TimeDistribute來使CNN模型可以加到LSTM模型一起使用，使用Conv2D方法只有CNN模型，並沒有時間序列，之後才加入LSTM。參數如下表: <br>

| Model         | CNN\_LSTM                                                                                                 | LRCN                                                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Architecture  | Sequential                                                                                                | Sequential                                                                                                         |
| First Layer   | ConvLSTM2D:<br>filters = 8,<br>kernel\_size = (5, 5)<br>activation = “relu”,<br>return\_sequences = True  | TimeDistributed:<br>Conv2D:<br>filters = 8,<br>kernel\_size = (5, 5),<br>padding = “same”,<br>activation = “relu”  |
| Second Layer  | MaxPooling3D:<br>pool\_size = (1, 2, 2),<br>padding = “same”                                              | TimeDistributed:<br>MaxPooling2D:<br>pool\_size = (4, 4),<br>padding=“valid”                                       |
| Third Layer   | ConvLSTM2D:<br>filters = 16,<br>kernel\_size = (3, 3)<br>activation = “relu”,<br>return\_sequences = True | TimeDistributed:<br>Dropout(0.25)                                                                                  |
| Fourth Layer  | MaxPooling3D:<br>pool\_size = (1, 2, 2),<br>padding = “same”                                              | TimeDistributed:<br>Conv2D:<br>filters = 16,<br>kernel\_size = (3, 3),<br>padding = “same”,<br>activation = “relu” |
| Fifth Layer   | TimeDistributed:<br>Dropout(0.2)                                                                          | TimeDistributed:<br>MaxPooling2D:<br>pool\_size = (4, 4),<br>padding='valid'                                       |
| Sixth Layer   | ConvLSTM2D:<br>filters = 32,<br>kernel\_size = (3, 3)<br>activation = “relu”,<br>return\_sequences = True | TimeDistributed:<br>Conv2D:<br>filters = 32,<br>kernel\_size = (3, 3),<br>padding = “same”,<br>activation = “relu” |
| Seventh Layer | MaxPooling3D:<br>pool\_size = (1, 2, 2),<br>padding = “same”                                              | TimeDistributed:<br>MaxPooling2D:<br>pool\_size = (2, 2),<br>padding='valid'                                       |
| Eighth Layer  | Flatten                                                                                                   | TimeDistributed:<br>Flatten                                                                                        |
| Ninth Layer   | Dense:<br>units= length of classes number,<br>activation = “softmax”                                      | LSTM:<br>units = 32,<br>activation = “relu”                                                                        |
| Tenth Layer   |                                                                                                           | Dense:<br>units= length of classes number,<br>activation = “softmax”                                               |
