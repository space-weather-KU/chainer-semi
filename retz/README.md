RETZの使い方

大本のドキュメント　https://gist.github.com/kuenishi/2c6b505b34db6aa77e0f61314473a57c


# Windows編

## 1. Javaランタイムのインストール

Anaconda Prompt, もしくはコマンドプロンプトで`java　-version`と打ってみてください。これがjavaのバージョンを返す場合は、すでにjavaがインストールされています。とくにインストールの必要はありません。
Javaがなければ、こちらからインストールしてください。　https://www.java.com/ja/download/help/download_options.xml

## 2. Retz 本体の使用

Windows用に調整したUSBディスクを渡します。

- `retz-client.bat`が、仮想マシンを制御するスクリプトです。
このスクリプトが何をしているかは、`type retz-client.bat`や`notepad retz-client.bat`で、中身を確認してみてください。以下も同じです。
- `load-app.bat` で、仮想マシンを用意します。
- `load-app.bat` で、仮想マシンを用意します。
- `get-file.bat <プロセス番号>`　で、ファイルを取得します。
# 何ができるの？

# さらに先に進むには？
参考文献　https://arxiv.org/abs/1701.00160
