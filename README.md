# colorization
## データセット作成手順  

### データセットをダウンロード
-  [Places Database](http://places.csail.mit.edu/user/register.php)でユーザ登録する
-  登録後、[imagesPlaces205_resize.tar.gz](http://places.csail.mit.edu/download_places/imagesPlaces205_resize.tar.gz)(データサイズ140GB)をダウンロードする

### データセットを展開
-  ターミナルを起動
-  imagesPlaces205_resize.tar.gzを保存したディレクトリに移動  
`$ cd data_location`
-  以下のコマンドを実行  
`$ tar xzvf imagesPlaces205_resize.tar.gz`  
-  データセットが展開される

### 不要なデータを除去しHDF5ファイルを作成
-  プログラム内の以下のパラメータを自身の設定に変更する  
    -  data_location : データセットを展開したディレクトリのルートパス
    -  output_location : HDF5ファイルを保存するディレクトリのルートパス
    -  output_size : 保存したいデータセットの画像サイズ
    -  test_size : テストデータ数
-  create_raw_dataset.pyを実行
