# 宇宙天気予報研究Web-UI 操作マニュアル

まず、Web-UIのURL http://54.238.211.181:8080/hello にアクセス。



![](https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/web-UI/figure/hello.png)

ユーザーとパスワードを聞かれるので入力する。すると、以下のようなソースコードを投稿する画面になる。

![](https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/web-UI/figure/submit.png)

- Job nameは、このジョブを区別できる分かりやすい名前を英数字を使って指定する(32文字まで有効)。
- time limitは1時間単位で、最大24時間まで指定できる。
- select queue は、`S` または`G`を指定できる。`Service Utilization`を参考に、空いてそうな方を指定すると良い。
- priority は、高い数字を指定するほど優先して実行される。
- ソースコード欄には実行したいソースコードを記入する。既存のサンプル集 https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/kit/ のなかにあるソースコードや、すでに[Files](http://54.238.211.181/) 以下に投稿されて実験が成功しているフォルダの中から、main.pyは、ここに投稿できる形になっている。

![](https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/web-UI/figure/submit-code.png)

Submitボタンを押すと以下の画面に移動する。

![](https://github.com/nushio3/tsubacloume/raw/master/figure/submitted.png)

現在のジョブの実行状況や

![](https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/web-UI/figure/job_status.png)

ジョブの生成したファイルの一覧が見れる。

![](https://github.com/space-weather-KU/chainer-semi/raw/master/learn-sun/web-UI/figure/files.png)


# Q&A

## Q. 何を持って予報機の完成ないしは改善とするのでしょうか？

A. 「何を持って予報機の完成ないしは改善」とするのか、については、最終的に
はフレア予測の的中率やTSSなどの、宇宙天気的指標をもってくることになり
ますが、いまはその前段階として、「観測データだけからいかに未来予測動画
を生成できるか、そもそも出来るのか？」を試しています。

ですので、現時点では、「予想動画のぼやけが取れれ」ばいい、つまり、
adversarial networkのちからを利用して、リアリスティックな動画を生成で
きるか、ということを試していければと思います。この時点では、定量的な指
標を立てられないので（もし指標があるなら、それを直接最適化すればいい）
関様自身をはじめ太陽を見慣れた人のちからが必要です。

評価指標をどうするのか？というのは、「より自然な画像を生成する」ことを
目標とするadversarial networkの研究に共通の未解決課題です。


もし定量的な指標が必要であれば、動画予測でなく、画像からGoesフラックス予測のほうを試してみてください。

https://github.com/space-weather-KU/chainer-semi/blob/master/learn-sun/kit/01-goes-flux-predictor.py

たとえば、以下の実行結果を参照ください。Goesフラックスの予想結果のグラフや、予測の平均誤差を出力します。

http://54.238.211.181/2017/02/11/72h_94_J8YkjsJvN6wc/
http://54.238.211.181/2017/02/11/72h_94_J8YkjsJvN6wc/prediction-history.png
http://54.238.211.181/2017/02/11/72h_94_J8YkjsJvN6wc/stdout.txt


## Q. どのようなパラメータを変えられるのか？

https://github.com/space-weather-KU/chainer-semi/blob/master/learn-sun/kit/06-hmi.py

を例にとって説明します。

・35行目　image_wavelengths = [94,211,'hmi']
の部分のリストを変えると、使用する画像の波長の一覧を変えられます。

選べるのは 94, 131, 171, 193, 211, 304, 'hmi' です。

・37行目からの　optimizer_p = chainer.optimizers.SMORMS3()

は、predictor, generator, discriminatorに使うoptimizerを指定しています。

利用可能なOptimizerの一覧については、Chainerのマニュアルを参照ください。

http://docs.chainer.org/en/stable/reference/optimizers.html?highlight=optimizers



・46行目からの　Mp=6, Mg=6, Md=6
を増やすほど、それぞれpredictor, generator, discriminatorのニューロンが多くなります。
学習は遅くなりますが、学べることは増えます。


・60行目からの関数　`get_normalized_image_variable`　は、画像を
どのように規格化してニューラルネットワークに見せるかを指定している関数です。

たとえば、ret = F.sigmoid(x / 300)の300の部分
これは、明るさが300以下のピクセルについてはおよそ線形に、
明るさが300をうわまわるピクセルに関しては明るさが飽和するように加工して、ニューラルネットワークに見せることを意味します。
いわば、300がニューラルネットワークに見せたい典型的な明るさとなります。




## Q. 磁場画像を扱えないか？

A. 我々のシステムでHMI画像を読み込めるようにし、河村案のHMI-94-211の組
み合わせで予測器をつくってみました。

http://54.238.211.181/2017/02/24/HMI_94_211_test9FckLL5UY9Na/
http://54.238.211.181/2017/02/24/HMI_94_211_miniYSIsm4y4S1MG/

ご参考になれば幸いです！
