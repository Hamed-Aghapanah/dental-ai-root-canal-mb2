import torch

ckpt_path = r"D:\project\0000_OPG\lengani\Unet_run\unet_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Best epoch:", ckpt["epoch"])
print("Validation metrics:", ckpt["val"])


# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# ckpt_path = r"D:\project\0000_OPG\lengani\Unet_run\unet_best.pt"
# ckpt = torch.load(ckpt_path, map_location=device)

# model = UNet(
#     in_ch=3,
#     out_ch=1,
#     base=ckpt.get("base", 32),
#     drop=ckpt.get("dropout", 0.20)
# ).to(device)

# model.load_state_dict(ckpt["model"])
# model.eval()

# print("Loaded best model from epoch:", ckpt["epoch"])
# print(ckpt["val"])