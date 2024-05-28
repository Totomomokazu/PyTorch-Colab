# パッケージの確認
# インストール済みのパッケージを全て表示します。
# PyTorch（torch）がインストールされていることを確認しましょう。


# !pip list
     
# Tensorの生成
# torchのtensor関数によりTensorを生成します。
# 以下のセルではPythonのリストからTensorを生成します。
# また、type( )により型を確認します。


# import torch
# a = torch.tensor([1,2,3])
# print(a, type(a))
     
# 他にも、様々な方法でTensorを生成することができます。


# 2重のリストから生成
b = torch.tensor([[1, 2],
                  [3, 4]])
print(b)

# dypeを指定し、倍精度のTensorにする
c = torch.tensor([[1, 2],
                  [3, 4]], dtype=torch.float64)
print(c)

# 0から9までの数値で初期化
d = torch.arange(0, 10)
print(d)

# すべての値が0の、2×3のTensor
e = torch.zeros(2, 3)
print(e)

# すべての値が乱数の、2×3のTensor
f = torch.rand(2, 3)
print(f)

# Tensorの形状はsizeメソッドで取得
print(f.size())
     
# TensorとNumPyの配列の変換
# numpy()メソッドでTensorをNumPyの配列に変換することができます。
# また、from_numpy( )関数でNumPyの配列をTensorに変換することができます。


# Tensor → NumPy
a = torch.tensor([[1, 2],
                  [3, 4.]])
b = a.numpy()
print(b)

# NumPy → Tensor
c = torch.from_numpy(b)
print(c)
     
# 範囲を指定してアクセス
# 様々な方法で、Tensorの要素に範囲を指定してアクセスすることができます。


a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# 2つのインデックスを指定
print(a[0, 1])

# 範囲を指定
print(a[1:2, :2])

# リストで複数のインデックスを指定
print(a[:, [0, 2]])

# 3より大きい要素のみを指定
print(a[a>3])

# 要素の変更
a[0, 2] = 11
print(a)

# 要素の一括変更
a[:, 1] = 22
print(a)

# 10より大きい要素のみ変更
a[a>10] = 33
print(a)
     
Tensorの演算
Tensorによりベクトルや行列を表現することができます。
これらの演算は、一定のルールに基づき行われます。


# ベクトル
a = torch.tensor([1, 2, 3]) 
b = torch.tensor([4, 5, 6])

# 行列
c = torch.tensor([[6, 5, 4],
                  [3, 2, 1]])

# ベクトルとスカラーの演算
print(a + 3)

# ベクトル同士の演算
print(a + b) 

# 行列とスカラーの演算
print(c + 2)

# 行列とベクトルの演算（ブロードキャスト）
print(c + a)

# 行列同士の演算
print(c + c)
     
様々な値の計算
平均値、合計値、最大値、最小値など様々な値を計算する関数とメソッドが用意されています。


a = torch.tensor([[1, 2, 3],
                  [4, 5, 6.]])

# 平均値を求める関数
m = torch.mean(a)
print(m.item())  # item()で値を取り出す

# 平均値を求めるメソッド
m = a.mean()
print(m.item())

# 列ごとの平均値
print(a.mean(0))

# 合計値
print(torch.sum(a).item())

# 最大値
print(torch.max(a).item())

# 最小値
print(torch.min(a).item())
     