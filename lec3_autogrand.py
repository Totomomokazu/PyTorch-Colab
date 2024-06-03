# 自動微分の開始
import torch

x = torch.ones(2, 3, requires_grad=True)
print(x)


# Tensorの演算と自動微分
y = x + 2
print(y)
print(y.grad_fn)

z = y * 3
print(z)

out = z.mean()
print(out)


# 勾配の計算
a = torch.tensor([1.0], requires_grad=True)
b = a * 2  # bの変化量はaの2倍
b.backward()  # 逆伝播
print(a.grad)  # aの勾配（aの変化に対するbの変化の割合）

def calc(a):
    b = a*2 + 1
    c = b*b 
    d = c/(c + 2)
    e = d.mean()
    return e

x = [1.0, 2.0, 3.0]
x = torch.tensor(x, requires_grad=True)
y = calc(x)
y.backward()
print(x.grad.tolist())  # xの勾配（xの各値の変化に対するyの変化の割合）

delta = 0.001  #xの微小変化

x = [1.0, 2.0, 3.0]
x = torch.tensor(x, requires_grad=True)
y = calc(x).item()

x_1 = [1.0+delta, 2.0, 3.0]
x_1 = torch.tensor(x_1, requires_grad=True)
y_1 = calc(x_1).item()

x_2 = [1.0, 2.0+delta, 3.0]
x_2 = torch.tensor(x_2, requires_grad=True)
y_2 = calc(x_2).item()

x_3 = [1.0, 2.0, 3.0+delta]
x_3 = torch.tensor(x_3, requires_grad=True)
y_3 = calc(x_3).item()

# 勾配の計算
grad_1 = (y_1 - y) / delta
grad_2 = (y_2 - y) / delta
grad_3 = (y_3 - y) / delta

print(grad_1, grad_2, grad_3)