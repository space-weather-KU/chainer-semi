# 宇宙天気予報研究Web-UI 操作マニュアル

まず、Web-UIのURL http://54.238.211.181:8080/hello にアクセス。



![](https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/web-UI/figure/hello.png)

ユーザーとパスワードを聞かれるので入力する。すると、以下のようなソースコードを投稿する画面になる。

![](https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/web-UI/figure/submit.png)

- Job nameは、このジョブを区別できる分かりやすい名前を英数字を使って指定する(32文字まで有効)。
- time limitは1時間単位で、最大24時刊まで指定できる。
- ソースコード欄には実行したいソースコードを記入する。既存のサンプル集 https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/kit/ にあるソースコードや、すでに[Files](http://54.238.211.181/) 以下に投稿されているフォルダのmain.pyを投稿できる。

![](https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/web-UI/figure/submit-code.png)

Submitボタンを押すと以下の画面に移動する。

![](https://github.com/nushio3/tsubacloume/raw/master/figure/submitted.png)

現在のジョブの実行状況や

![](https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/web-UI/figure/job status.png)

ジョブの生成したファイルの一覧が見れる。

![](https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/web-UI/figure/files.png)
