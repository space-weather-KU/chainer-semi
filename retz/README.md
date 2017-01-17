# RETZとは？

公式サイト　https://github.com/retz/retz

仮想化したコンピュータをコントロールし、プログラムを予約して実行したり、結果を取得したりするためのソフトウェアです。

RETZの公式ドキュメント　https://gist.github.com/kuenishi/2c6b505b34db6aa77e0f61314473a57c

# 使い方
retz-clientをインストールした状態で、｀retz-client｀とだけタイプすると、以下のように、コマンド一覧を表示してくれます。

```
$ retz-client
ERROR Invalid subcommand
INFO Subcommands:
INFO 	help	Print help ('-s <subcommand>' see detail options) (io.github.retz.cli.CommandHelp)
INFO 	config	Check configuration file (io.github.retz.cli.CommandConfig)
INFO 	list	list all jobs (io.github.retz.cli.CommandList)
INFO 	schedule	Schedule a job (io.github.retz.cli.CommandSchedule)
INFO 	get-job	Get details of a job (io.github.retz.cli.CommandGetJob)
INFO 	get-file	Get file from sandbox of a job (io.github.retz.cli.CommandGetFile)
INFO 	list-files	List files in the sandbox of a job (io.github.retz.cli.CommandListFiles)
INFO 	kill	Kill a job (io.github.retz.cli.CommandKill)
INFO 	killall	Kill a group of jobs (io.github.retz.cli.CommandKillall)
INFO 	run	Schedule and watch a job (io.github.retz.cli.CommandRun)
INFO 	get-app	Get an application details (io.github.retz.cli.CommandGetApp)
INFO 	list-app	List all loaded applications (io.github.retz.cli.CommandListApp)
INFO 	load-app	Load an applitation that you want to run (io.github.retz.cli.CommandLoadApp)
```

基本的に、｀retz-client <コマンド>｀の形で使います。

各コマンドの詳しい解説はこちらにあります　https://github.com/retz/retz/blob/master/doc/api.rst#client-cli-and-api

また、` retz-client help -s <コマンド>`とすると、コマンドごとのヘルプを見ることができます。

```
$ retz-client help -s list
INFO Usage: list(list all jobs) [options]
  Options:
    --state
      State of jobs
      Default: QUEUED
      Possible Values: [CREATED, QUEUED, STARTING, STARTED, FINISHED, KILLED]
    --tag
      Tag name to show
```

## 設定

`retz-client`はJavaで書かれています。ですので、Javaが走る環境であれば、`retz-client-*.*.*-all.jar`さえあれば使えます(`*`はバージョン情報です。)

クライアントの接続先などの大切な認証情報は、`retz.properties`というファイルで管理しています。このファイルはみなさん専用のものを個別に配布しますので、他人に見られないように保管して下さい。

`retz-client-*.jar -C retz.properties`


## コマンド紹介

これらのコマンド例は、 https://github.com/space-weather-KU/chainer-semi/tree/master/retz に`*.sh`というファイル名で保存してあります。

## load-app

仮想マシンはdockerを使って作れます。

- dockerのチュートリアル https://docs.docker.com/engine/getstarted/
- 今回用意した仮想マシンイメージ https://hub.docker.com/r/nushio3/chainer-semi/

load-appで、仮想マシンイメージを指定して「アプリケーション」を作ります。

なお,`\`は長い行を区切る記号で、実際には`\`を抜いて、一行で入力して下さい。

```
retz-client load-app -A movie-predict-nushio \
  --container docker --image nushio3/chainer-semi \
  -F https://raw.githubusercontent.com/space-weather-KU/chainer-semi/master/learn-sun/nushio3/13-simple-moviepredict.py \
  --user root
```

- `-A` アプリケーション名を指定
- `--image` dockerhubに存在する仮想マシンイメージを紹介
- `-F` 起動時に読み込んでおくファイルのURLを指定
- `--mem` メモリ量を指定(単位はMB)
- そのほか、`retz-client help -s load-app`を参照！


## run

指定したアプリケーションを実行します。

```
retz-client run -A movie-predict-nushio -c 'apt-get update && apt-get -y install zip && python 13-simple-moviepredict.py' --mem 6000 --cpu 1 --stderr
```

- `-A` アプリケーション名を指定
- `-c` 起動時に実行するコマンドを指定
- `--mem` メモリ量を指定(単位はMB)
- そのほか、`retz-client help -s run`を参照！

## list

仮想マシンの一覧をstateごとに表示します。たとえば、現在実行中(`STARTED`)のマシンの一覧を表示するには次のようにします。

この表の2行目に表示されているのがタスク番号です。以降、仮想マシンの操作にはこのタスク番号を指定します。

```
$ retz-client list --state STARTED
WARN DANGER ZONE: TLS certificate check is disabled. Set 'retz.tls.insecure = false' at config file to supress this message.
INFO TaskId State   AppName              Command                                                                      Result Duration Scheduled                     Started                       Finished Tags
INFO 49     STARTED movie-predict-nushio apt-get update && apt-get -y install zip && python 13-simple-moviepredict.py -      -        2017-01-15T16:25:38.941+09:00 2017-01-15T16:25:48.762+09:00 N/A
```

## list-files

仮想マシン上のファイルの一覧を取得します。`retz-client list-files -i <タスク番号>`

```
$ retz-client list-files -i 49
WARN DANGER ZONE: TLS certificate check is disabled. Set 'retz.tls.insecure = false' at config file to supress this message.
INFO gid  mode       uid  mtime               size    path
INFO root -rw-r--r-- root 2017-01-15 16:25:48 2491    01-get-sun-image.py
INFO root -rw-r--r-- root 2017-01-15 16:25:48 5499    13-simple-moviepredict.py
INFO root drwxrwxr-x root 2016-12-15 13:13:33 4096    chainer-1.19.0
INFO root -rw-r--r-- root 2017-01-17 07:22:23 379827  image-input.png
INFO root -rw-r--r-- root 2017-01-17 07:22:23 380600  image-observed.png
INFO root -rw-r--r-- root 2017-01-17 07:22:25 274479  image-predict-0.png
INFO root -rw-r--r-- root 2017-01-17 07:22:27 268710  image-predict-1.png
INFO root -rw-r--r-- root 2017-01-17 07:22:28 264928  image-predict-2.png
INFO root -rw-r--r-- root 2017-01-17 07:22:30 262774  image-predict-3.png
INFO root -rw-r--r-- root 2017-01-17 07:22:31 261907  image-predict-4.png
INFO root -rw-r--r-- root 2017-01-17 07:22:33 260779  image-predict-5.png
INFO root -rw-r--r-- root 2017-01-17 07:22:33 2337332 images.zip
INFO root -rw-r--r-- root 2017-01-17 07:22:33 3157450 images.zip.base64
INFO root -rw-r--r-- root 2017-01-17 10:56:41 3218892 stderr
INFO root -rw-r--r-- root 2017-01-17 13:49:37 3870208 stdout
INFO root -rw-r--r-- root 2017-01-17 07:23:58 736541  sun-predictor-211-4hr.model
INFO root -rw-r--r-- root 2017-01-15 16:25:47 2109252 v1.19.0.tar.gz
```

## get-file

仮想マシンから指定したファイル名のファイルを取得します。

```
$ retz-client get-file -i 49 --binary --path images.zip -R .
WARN DANGER ZONE: TLS certificate check is disabled. Set 'retz.tls.insecure = false' at config file to supress this message.
INFO Saving images.zip to ./images.zip
```

# Windows編

## 1. Javaランタイムのインストール

Anaconda Prompt, もしくはコマンドプロンプトで`java　-version`と打ってみてください。これがjavaのバージョンを返す場合は、すでにjavaがインストールされています。とくにインストールの必要はありません。
Javaがなければ、こちらからインストールしてください。　https://www.java.com/ja/download/help/download_options.xml

## 2. Retz 本体の使用

Windows用に調整したUSBディスクを渡します。

- `retz-client.bat`が、仮想マシンを制御するスクリプトです。基本的には、`retz-client.bat`を`retz-client`の代わりに使えば大丈夫です。
このスクリプトが何をしているかは、`type retz-client.bat`や`notepad retz-client.bat`で、中身を確認してみてください。以下も同じです。
- `load-app.bat` で、仮想マシンを用意します。
- `load-app.bat` で、仮想マシンを用意します。
- `get-file.bat <プロセス番号>`　で、ファイルを取得します。

# 何ができるの？

上記のステップで、太陽活動を予測し、太陽が回転しているようにみえる動画が作れるはずです。

# さらに先に進むには？

ここまでは、画像の類似度を単にピクセルごとの色の二乗誤差で判定していました。
そのため、どうしても、ある程度ボケた画像が生成されてしまっていました。

この次の段階に進むには、「太陽画像っぽさ」を判定する「Adversarial Network」を作り、
判定者に「太陽画像っぽい」と判定してもらえるよう最適な画像を生成するという、Generative Adversarial Network
という技術が大変有効です。

ちょうど、Generative Adversarial Networkに関するチュートリアルが公開されました。 https://arxiv.org/abs/1701.00160
("NIPS 2016 Tutorial: Generative Adversarial Networks" Ian Goodfellow)次はこれをみんなで読みましょう。

二乗誤差とAdversarial Networkの性質の違いを端的に理解するには,画像着色の例が分かりやすいです。
- この論文の図1をご覧ください http://cs231n.stanford.edu/reports2016/224_Report.pdf ("Automatic Colorization with Deep Convolutional Generative Adversarial
Networks", Stephen Koo)
- または、題材がアニメ寄りですが、この記事をご覧ください http://qiita.com/taizan/items/cf77fd37ec3a0bef5d9d
