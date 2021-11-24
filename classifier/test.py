import os

PATH = 'trained_model'
# if not os.path.exists(PATH):
#     os.mkdir(PATH)
print(os.path.join(PATH, 'cifar_net.pth'))