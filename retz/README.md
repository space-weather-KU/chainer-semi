# RETZとは？

公式サイト　https://github.com/retz/retz

仮想化したコンピュータをコントロールし、プログラムを予約して実行したり、結果を取得したりするためのソフトウェアです。

RETZの公式ドキュメント　https://gist.github.com/kuenishi/2c6b505b34db6aa77e0f61314473a57c

＃ 使い方
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
参考文献　https://arxiv.org/abs/1701.00160
