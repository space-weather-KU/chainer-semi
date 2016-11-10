# 参考書の場所
BBT宇宙天気予報研究会では、1日あたり１．５TBのペースで取得されている膨大な太陽観測データを利用して、太陽の未来を予測する手段を創り出そうとしています。そのための手段として、深層学習を使います。深層学習を学ぶために、Preferred Networks社が開発している深層学習ライブラリChainer,およびChainerがブラウザ上で学習できるChainer Playgroundを使います。


Chainer Playgroundは、1/1/2節を動かしてみたあと、1/2節をすべて攻略する、という順序でやれば、Chainerの仕組みが先に分かるので、おすすめです。


- 公式bookの1/1/2
- 公式bookの1/2/1


Chainer Playgroundはある程度のPythonの知識を前提としています。Pythonについてじっくり学びたい方は、Codecademy でPythonのコースを進めるのもよいです。 https://www.codecademy.com/learn/python


また、書籍で学びたい方は、一例ですが　柴田　淳 著「みんなのPython」　http://amzn.to/2ffyKJL　がおすすめです。




# プログラミング攻略法


いまから、皆様は深層学習という、人類の最先端の技術、かつ、多くの人がよってたかって開拓している真っ最中の分野を学ぼうとしています。ですので、分からない、ついていけない、と感じることがあっても決して自分を責めたり、くじけたりしないで下さい。私や、深層学習の研究者を含む大多数もそう感じています。今日最先端だった理論やツールは、1年後、いや数週間後にはもう陳腐化しているかもしれません。


ですからこのゼミでは、与えられたプログラムの使い方ではなく、プログラムの使い方の「調べ方」をなるべくお伝えしたいと思っています。このゼミで学んだことが、深層学習やPythonだけに限らず、皆様が今後プログラミングに向き合うための力になればと願っています。




# Python攻略法
最終的には公式のドキュメントが頼りになります
- Python2.7 ドキュメント http://docs.python.jp/2.7/index.html
- Chainer ドキュメント　http://docs.chainer.org/en/stable/


## ある変数の正体を調べたいとき： print, type, dir


print関数を使って、変数の値を調べることができます。
また、type(x)は、xの型を返す関数です。たとえば、次のプログラムを実行すると

```
x=4.2
print(x)
print(type(x))
```

次のような出力が帰ってきます。このことから、変数ｘの値は4.2で、その型は’float’であることがわかります。（floatは、pythonで実数を表現するのに通常つかわれる型です。）

```
4.2
<type 'float'>
```



さらに、dir(x)は、xの持っているメソッドを全て表示してくれます。

```
x="deep learning"
print(type(x))
print(dir(x))
```




```
<type 'str'>
['__add__', '__class__', '__contains__', '__delattr__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__getslice__', '__gt__', '__hash__', '__init__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_formatter_field_name_split', '_formatter_parser', 'capitalize', 'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']
```

ここから、xは’str’型(文字列型)であること、またcapitalize, split, replaceなどのメソッドを持っていることがわかります。


メソッドや型の名前がわかれば、その使い方は「python str capitalize [検索]」などとして調べることができます。さっそく使ってみましょう。

```
x = ”deep learning”
print(x.capitalize())
print(x.split(" "))
print(x.replace("e","o"))
print(x.replace("e","n").split("n"))
```

は、次のような結果になります。

```
Deep learning
['deep', 'learning']
doop loarning
['d', '', 'p l', 'ar', 'i', 'g']
```

## 配列変数を調べたいとき： type, shape, dtype


