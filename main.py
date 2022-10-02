import torch
import librosa
"""
torch finds 2 devices: cuda 0 and cpu
ctrl op function om naar declaration te gaan
"""
x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
print(torch.has_cuda)

device = torch.device("cuda")
x = torch.ones(5,device = device)
y = torch.ones(5)
y = y.to(device)
z = x+y
z = z.to("cpu")
z = z.numpy()
print(x,y,z)

def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('PyCharm')

