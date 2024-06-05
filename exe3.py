# ### 1. データの準備
# - **データの収集**: 目的に合ったデータセットを用意する。
# - **データの前処理**: データをニューラルネットワークに入力できる形式に変換する（例: 正規化、テンソルへの変換、データ拡張）。
# - **データの分割**: データセットを訓練データとテストデータに分割する。

# ### 2. モデルの設計
# - **モデルの定義**: ニューラルネットワークのアーキテクチャを定義する。入力層、隠れ層、出力層の構成を決める。
# - **初期化**: モデルのパラメータを初期化する。

# ### 3. 損失関数と最適化手法の選択
# - **損失関数の定義**: モデルの予測と実際のラベルとの誤差を計算する関数を選ぶ（例: 交差エントロピー、平均二乗誤差）。
# - **最適化手法の選択**: モデルのパラメータを更新するアルゴリズムを選ぶ（例: 確率的勾配降下法、Adam）。

# ### 4. 訓練プロセス
# - **フォワードパス**: 訓練データをモデルに入力し、予測を取得する。
# - **損失の計算**: 損失関数を使って予測と実際のラベルとの誤差を計算する。
# - **バックプロパゲーション**: 誤差を逆伝播させて各パラメータの勾配を計算する。
# - **パラメータの更新**: 勾配を使って最適化手法によりモデルのパラメータを更新する。
# - **繰り返し**: 上記のプロセスを指定したエポック数または収束するまで繰り返す。

# ### 5. モデルの評価
# - **評価モードの設定**: モデルを評価モードに設定する。
# - **テストデータによる評価**: テストデータを使ってモデルの予測精度を計測する。
# - **評価指標の計算**: 正解率、損失、F1スコアなどの評価指標を計算する。

# ### 6. 結果の可視化と分析
# - **損失の推移のプロット**: 訓練中の損失の変化をグラフにプロットして視覚化する。
# - **評価結果の表示**: テストデータに対する評価結果を表示する。

# ### まとめ
# 1. **データの準備**: データの収集、前処理、分割。
# 2. **モデルの設計**: モデルの定義と初期化。
# 3. **損失関数と最適化手法の選択**。
# 4. **訓練プロセス**: フォワードパス、損失計算、バックプロパゲーション、パラメータ更新、繰り返し。
# 5. **モデルの評価**: テストデータによる評価と指標計算。
# 6. **結果の可視化と分析**: 損失の推移のプロット、評価結果の表示。






# ### 1. データの準備
# - **データの収集と前処理**
#   ```python
#   mnist_train = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
#   mnist_test = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
#   ```
#   - MNISTデータセットをダウンロードしてテンソル形式に変換しています。

# - **データの分割**
#   ```python
#   train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
#   test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
#   ```
#   - 訓練データとテストデータに分けて、ミニバッチとして読み込む設定をしています。

# ### 2. モデルの設計
# - **モデルの定義と初期化**
#   ```python
#   class Net(nn.Module):
#       def __init__(self):
#           super().__init__()
#           self.fc1 = nn.Linear(img_size*img_size, 1024)
#           self.fc2 = nn.Linear(1024, 512)
#           self.fc3 = nn.Linear(512, 10)
  
#       def forward(self, x):
#           x = x.view(-1, img_size*img_size)
#           x = F.relu(self.fc1(x))
#           x = F.relu(self.fc2(x))
#           x = self.fc3(x)
#           return x

#   net = Net()
#   net.cuda()  # GPU対応
#   print(net)
#   ```
#   - ネットワークのアーキテクチャを定義し、モデルをGPUに転送しています。

# ### 3. 損失関数と最適化手法の選択
# - **損失関数の定義**
#   ```python
#   loss_fnc = nn.CrossEntropyLoss()
#   ```
#   - 交差エントロピー誤差関数を選択しています。

# - **最適化手法の選択**
#   ```python
#   optimizer = optim.SGD(net.parameters(), lr=0.01)
#   ```
#   - 確率的勾配降下法（SGD）を使用して最適化を行う設定をしています。

# ### 4. 訓練プロセス
# - **訓練プロセス**
#   ```python
#   for i in range(10):  # 10エポック学習
#       net.train()  # 訓練モード
#       loss_train = 0
#       for j, (x, t) in enumerate(train_loader):  # ミニバッチ（x, t）を取り出す
#           x, t = x.cuda(), t.cuda()  # GPU対応
#           y = net(x)
#           loss = loss_fnc(y, t)
#           loss_train += loss.item()
#           optimizer.zero_grad()
#           loss.backward()
#           optimizer.step()
#       loss_train /= j+1
#       record_loss_train.append(loss_train)

#       net.eval()  # 評価モード
#       loss_test = 0
#       for j, (x, t) in enumerate(test_loader):  # ミニバッチ（x, t）を取り出す
#           x, t = x.cuda(), t.cuda()
#           y = net(x)
#           loss = loss_fnc(y, t)
#           loss_test += loss.item()
#       loss_test /= j+1
#       record_loss_test.append(loss_test)

#       if i%1 == 0:
#           print("Epoch:", i, "Loss_Train:", loss_train, "Loss_Test:", loss_test)
#   ```
#   - 訓練モードでフォワードパス、損失計算、バックプロパゲーション、パラメータ更新を行っています。
#   - 評価モードでテストデータを用いた損失計算を行っています。

# ### 5. モデルの評価
# - **評価プロセス**
#   ```python
#   correct = 0
#   total = 0
#   for i, (x, t) in enumerate(test_loader):
#       x, t = x.cuda(), t.cuda()  # GPU対応
#       x = x.view(-1, img_size*img_size)
#       y = net(x)
#       correct += (y.argmax(1) == t).sum().item()
#       total += len(x)
#   print("正解率:", str(correct/total*100) + "%")
#   ```
#   - テストデータを用いてモデルの予測精度を計測しています。

# ### 6. 結果の可視化と分析
# - **損失の推移のプロット**
#   ```python
#   import matplotlib.pyplot as plt

#   plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
#   plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
#   plt.legend()

#   plt.xlabel("Epochs")
#   plt.ylabel("Error")
#   plt.show()
#   ```
#   - 訓練損失とテスト損失の推移を可視化しています。
