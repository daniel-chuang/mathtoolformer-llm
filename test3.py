# import bitsandbytes as bnb
import torch

print(torch.__version__)
print(torch.cuda.is_available())

# # Create a dummy tensor and run an operation
# tensor = torch.randn(10, 10).cuda()
# result = bnb.optim.Adam(tensor, lr=0.01)
# print("Bitsandbytes Adam update successful")