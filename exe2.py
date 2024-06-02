# データを訓練用とテスト用に分割
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
# torchをインポートして、PyTorchの機能を使用できるようにします。
# datasetsをインポートして、Scikit-learnのデータセットを使用できるようにします。
# train_test_splitをインポートして、データセットを訓練用とテスト用に分割する機能を使用できるようにします。


digits_data = datasets.load_digits()
# 手書き数字データセットを読み込みます。このデータセットには0から9までの数字の画像が含まれています。

digit_images = digits_data.data
labels = digits_data.target
x_train, x_test, t_train, t_test = train_test_split(digit_images, labels)  # 25%がテスト用

# digit_imagesにデータセットの画像データ（特徴量）を代入します。
# labelsに画像データに対応するラベル（目標値）を代入します。
# train_test_splitを使って、データセットを訓練データ（75%）とテストデータ（25%）に分割します。random_state=42は分割の再現性を確保するための乱数シードです。



# Tensorに変換
x_train = torch.tensor(x_train, dtype=torch.float32)
t_train = torch.tensor(t_train, dtype=torch.int64) 
x_test = torch.tensor(x_test, dtype=torch.float32)
t_test = torch.tensor(t_test, dtype=torch.int64) 
# # 訓練データとテストデータをPyTorchのTensorに変換します。
# 分割した訓練データとテストデータをPyTorchのテンソルに変換します。dtype=torch.float32は特徴量を32ビット浮動小数点数に変換し、dtype=torch.int64はラベルを64ビット整数に変換します。




# モデルの構築
# 3層のニューラルネットワーク（64-128-64-10）を構築します。
from torch import nn
# # PyTorchのニューラルネットワークモジュール（nn）をインポートします。

net = nn.Sequential(
    nn.Linear(64,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10)
    )
# nn.Sequentialを使って、順番に層を積み重ねたニューラルネットワークを定義します。
# nn.Linear(64, 128)は、入力ユニット数64、出力ユニット数128の全結合層を定義します。
# nn.ReLU()は、活性化関数ReLU（Rectified Linear Unit）を定義します。
# nn.Linear(128, 64)は、入力ユニット数128、出力ユニット数64の全結合層を定義します。
# nn.ReLU()は、再び活性化関数ReLUを定義します。
# nn.Linear(64, 10)は、入力ユニット数64、出力ユニット数10（0から9までの10クラス）の全結合層を定義します。

print(net)
# 定義したニューラルネットワークの構造を表示します。

from torch import optim
# PyTorchの最適化アルゴリズムモジュール（optim）をインポートします。
# 交差エントロピー誤差関数
loss_fnc = nn.CrossEntropyLoss()
# 交差エントロピー誤差関数を定義します。この誤差関数は分類問題においてよく使われます。


# 最適化アルゴリズム
optimizer = optim.SGD(net.parameters(), lr=0.01)
# 確率的勾配降下法（SGD）を最適化アルゴリズムとして定義します。lr=0.01は学習率です。


# 損失のログ
record_loss_train = []
record_loss_test = []
# 訓練データとテストデータの損失を記録するためのリストを初期化します。


# 1000エポック学習
for i in range(1000):
    # 1000エポックの学習ループを開始します。

    # 勾配を0にリセットします。これはパラメータの更新を行う前に行う必要があります。
    optimizer.zero_grad()
    
    # 順伝播
    y_train = net(x_train)
    y_test = net(x_test)
    # 訓練データとテストデータをニューラルネットワークに入力し、出力（予測値）を計算します。
    
    # 誤差を求める
    loss_train = loss_fnc(y_train, t_train)
    loss_test = loss_fnc(y_test, t_test)


    record_loss_train.append(loss_train.item())
    record_loss_test.append(loss_test.item())
    # 訓練データとテストデータの損失をリストに記録します。

    # 逆伝播（勾配を求める）
    loss_train.backward()
    # 訓練データの損失に対して逆伝播を行い、勾配を計算します。

    
    # パラメータの更新
    optimizer.step()
    # 最適化アルゴリズムを使ってパラメータを更新します。

    if i%100 == 0:
        print("Epoch:", i, "Loss_Train:", loss_train.item(), "Loss_Test:", loss_test.item())
        # 100エポックごとに訓練データとテストデータの損失を表示します。


#  1000エポックにわたり、モデルを訓練し、損失を記録します。



# 誤差の推移
import matplotlib.pyplot as plt
# グラフ描画ライブラリMatplotlibをインポートします。

plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
plt.legend()
# 訓練データとテストデータの損失の推移をプロットし、凡例を追加します。


plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
# 訓練データとテストデータの損失の推移をプロットします。



# 正解率
y_test = net(x_test)
count = (y_test.argmax(1) == t_test).sum().item()
print("正解率:", str(count/len(y_test)*100) + "%")
# テストデータに対するモデルの正解率を計算します。