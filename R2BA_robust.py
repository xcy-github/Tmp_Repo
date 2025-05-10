import numpy as np

from ImageNet_Model import (
    device, get_FatFormer_model, get_ResNetND_model, get_UnivFD_model, get_DRCT_model, get_CNNSpot_model
)
from io import BytesIO
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms

cpu_num = 5  # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)


class CustomImageDataset(Dataset):
    def __init__(self, file_paths, transform=None, robust="JPEG"):
        self.file_paths = file_paths  # 存储文件路径列表
        self.transform = transform  # 预处理方法
        self.robust = robust  # 预处理方法

    def __len__(self):
        return len(self.file_paths)  # 返回数据集大小

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]  # 获取当前索引的图像路径
        image = Image.open(img_path).convert('RGB')  # 打开图像文件
        if self.robust == "JPEG":
            image = JPEG_robust(image)
        if self.robust == "GN":
            image = apply_gaussian_noise(image)
        if self.robust == "RT":
            image = F.rotate(image, Rotate_Value)
        if self.robust == "RS":
            width, height = image.size
            transform_1 = transforms.Resize(width*Rs_scale)
            transform_2 = transforms.Resize(width)
            image = transform_1(image)
            image = transform_2(image)
        if self.transform:  # 如果有预处理方法，则应用
            image = self.transform(image)
        return image


def data_load_function(trans, img_dir, robust):
    file_paths = [x.path for x in os.scandir(img_dir)]
    dataset = CustomImageDataset(file_paths=file_paths, transform=trans, robust=robust)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader, file_paths


def test_other_model(img_ori_root_path, img_save_root_path, txt_path, robust):
    for i_test_model in test_model2attack:
        if i_test_model == 0:
            test_model, test_trans, test_class = get_CNNSpot_model()
        elif i_test_model == 1:
            test_model, test_trans, test_class = get_ResNetND_model()
        elif i_test_model == 2:
            test_model, test_trans, test_class = get_UnivFD_model()
        elif i_test_model == 3:
            test_model, test_trans, test_class = get_FatFormer_model()
        elif i_test_model == 4 or i_test_model == 5:
            test_model, test_trans, test_class = get_DRCT_model(choose_ckpt, MODEL_CHOICE[i_test_model])
        else:
            test_model, test_trans, test_class = None, None, None
        ori_dataloader, _ = data_load_function(test_trans, img_ori_root_path, robust=robust)
        adv_dataloader, _ = data_load_function(test_trans, img_save_root_path, robust=robust)
        res = [0, 0]
        for img, adv_img in zip(ori_dataloader, adv_dataloader):
            img = img.to(device)
            adv_img = adv_img.to(device)
            if test_class == 2:
                y_ori = test_model(img).softmax(dim=1)[:, -1:].item()
                y_adv = test_model(adv_img).softmax(dim=1)[:, -1:].item()
            else:
                y_ori = test_model(img).sigmoid().item()
                y_adv = test_model(adv_img).sigmoid().item()
            res[0] += 1 if y_ori > 0.5 else 0  # 识别错误的图像
            res[1] += 1 if y_ori > 0.5 and y_adv > 0.5 else 0  # 识别错误，但是攻击成功的数目
        res.append(round((res[0] - res[1]) / res[0], 6))
        with open(txt_path, "a") as print_res:
            print(f"Robust == {MODEL_CHOICE[i_test_model]}[ori:adv:sr]: {res}, "
                  f"total_num: {len(ori_dataloader)}", file=print_res)


def JPEG_robust(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=QUALITY)
    compressed_image = Image.open(img_byte_arr)
    return compressed_image


def apply_gaussian_noise(image, mean=0):
    img_array = np.array(image)
    sigma = GN_SIGMA ** 0.5
    noise = np.random.normal(mean, sigma, img_array.shape)
    noisy_image = img_array + noise * 255  # 调整噪声幅度
    return Image.fromarray(np.clip(noisy_image, 0, 255).astype('uint8'))


def test_robust(ORI_IMG_PATH, ADV_IMG_PATH, test_model_i, robust="JPEG"):
    if test_model_i == 0:
        test_model, test_trans, test_class = get_CNNSpot_model()
    elif test_model_i == 1:
        test_model, test_trans, test_class = get_ResNetND_model()
    elif test_model_i == 2:
        test_model, test_trans, test_class = get_UnivFD_model()
    elif test_model_i == 3:
        test_model, test_trans, test_class = get_FatFormer_model()
    elif test_model_i == 4 or test_model_i == 5:
        test_model, test_trans, test_class = get_DRCT_model(choose_ckpt, MODEL_CHOICE[test_model_i])
    else:
        test_model, test_trans, test_class = None, None, None
    print("------------------------------------")
    print(ORI_IMG_PATH)
    print(ADV_IMG_PATH)
    ori_dataloader, _ = data_load_function(test_trans, ORI_IMG_PATH, robust=robust)
    adv_dataloader, _ = data_load_function(test_trans, ADV_IMG_PATH, robust=robust)
    res = [0, 0]
    for img, adv_img in zip(ori_dataloader, adv_dataloader):
        img = img.to(device)
        adv_img = adv_img.to(device)
        if test_class == 2:
            y_ori = test_model(img).softmax(dim=1)[:, -1:].item()
            y_adv = test_model(adv_img).softmax(dim=1)[:, -1:].item()
        else:
            y_ori = test_model(img).sigmoid().item()
            y_adv = test_model(adv_img).sigmoid().item()
        res[0] += 1 if y_ori > 0.5 else 0  # 识别错误的图像
        res[1] += 1 if y_ori > 0.5 and y_adv > 0.5 else 0  # 识别错误，但是攻击成功的数目
    res.append(round((res[0] - res[1]) / res[0], 6))
    with open(TXT_PATH, "a") as print_res:
        print(f"{method_ATTACK}/{MODEL_CHOICE[test_model_i]}  res:{res}", file=print_res)


if __name__ == "__main__":
    RESULT_PATH = "/"
    DATASET_PATH = [["/DATASET/ImageNet/ProGAN/", "ImageNet/ProGAN"],
                    ["/DATASET/GenImage/VQDM/", "GenImage/VQDM"],
                    ["/DATASET/ImageNet/ldm_200", "ImageNet/ldm_200"]]
    MODEL_CHOICE = ["CNNSpot", "ResNetND", "UnivFD", "FatFormer", "UnivFD-Diff", "ConvB-Diff"]
    choose_ckpt = "DRCT"
    METHOD_CHOICE = [
        f"Black/{DATASET_PATH[0][1]}/Square",
        f"Black/{DATASET_PATH[0][1]}/SimBA",
        f"Black/{DATASET_PATH[0][1]}/BruSLi-3000",
        f"Ours/{DATASET_PATH[0][1]}/PSO_SSIM",
        f"White/{DATASET_PATH[0][1]}/FGSM",
        f"White/{DATASET_PATH[0][1]}/MIFGSM",
        f"White/{DATASET_PATH[0][1]}/PGD",
        f"White/{DATASET_PATH[0][1]}/Evade",

        f"Black/{DATASET_PATH[1][1]}/Square",
        f"Black/{DATASET_PATH[1][1]}/SimBA",
        f"Black/{DATASET_PATH[1][1]}/BruSLi-3000",
        f"Ours/{DATASET_PATH[1][1]}/PSO_SSIM",
        f"White/{DATASET_PATH[1][1]}/FGSM",
        f"White/{DATASET_PATH[1][1]}/MIFGSM",
        f"White/{DATASET_PATH[1][1]}/PGD",
        f"White/{DATASET_PATH[1][1]}/Evade",

        f"Black/{DATASET_PATH[2][1]}/Square",
        f"Black/{DATASET_PATH[2][1]}/SimBA",
        f"Black/{DATASET_PATH[2][1]}/BruSLi-3000",
        f"Ours/{DATASET_PATH[2][1]}/PSO_SSIM",
        f"White/{DATASET_PATH[2][1]}/FGSM",
        f"White/{DATASET_PATH[2][1]}/MIFGSM",
        f"White/{DATASET_PATH[2][1]}/PGD",
        f"White/{DATASET_PATH[2][1]}/Evade"
    ]

    METHOD_CHOICE_OLD = [
        f"Black/{DATASET_PATH[0][1]}/BruSLi-3000",
        f"Black/{DATASET_PATH[1][1]}/BruSLi-3000",
        f"Black/{DATASET_PATH[2][1]}/BruSLi-3000"
    ]

    # jpeg
    # TXT_PATH = "/jpeg_robust.txt"
    Q_list = [50, 60, 70, 80, 90]
    QUALITY = Q_list[4]
    # GN
    # TXT_PATH = "/gn_robust.txt"
    GN_ls = [(1/255)*(1/255), (2/255)*(2/255), (3/255)*(3/255), (4/255)*(4/255), (5/255)*(5/255)]
    GN_SIGMA = GN_ls[4]
    # RT
    TXT_PATH = "/RT_robust.txt"
    Rotate_ls = [2, 4, 6, 8, 10]
    Rotate_Value = Rotate_ls[4]  # ----------------------
    # RS
    # TXT_PATH = "/RS_robust.txt"
    Resize_ls = [1/2, 3/4, 5/4, 3/2, 7/4]
    Rs_scale = Resize_ls[0]

    ROBUST_MODE = "RT"  # JPEG
    with open(TXT_PATH, "a") as print_res:
        print(f"=== {ROBUST_MODE} ==={QUALITY,GN_SIGMA,Rotate_Value,Rs_scale} "
              f"RESULT:{RESULT_PATH}", file=print_res)

    choose_data = 0  # 前六个对应ImageNet/ProGAN
    dataset_ATTACK, save_ATTACK = DATASET_PATH[choose_data][0], DATASET_PATH[choose_data][1]
    test_model2attack = [0, 1, 2, 3]
    METHOD_CHOICE_now = METHOD_CHOICE[:8]
    for choose_method in range(len(METHOD_CHOICE_now)):
        method_ATTACK = METHOD_CHOICE_now[choose_method]  # 选择攻击方法
        for test_model_i in test_model2attack:
            ORI_IMG_PATH = dataset_ATTACK  # 原始
            ADV_IMG_PATH = f"{RESULT_PATH}{method_ATTACK}/{MODEL_CHOICE[test_model_i]}/"  # 对抗
            test_robust(ORI_IMG_PATH, ADV_IMG_PATH, test_model_i, robust=ROBUST_MODE)

    choose_data = 1  # 前六个对应ImageNet/ProGAN
    dataset_ATTACK, save_ATTACK = DATASET_PATH[choose_data][0], DATASET_PATH[choose_data][1]
    test_model2attack = [3, 4, 5]
    METHOD_CHOICE_now = METHOD_CHOICE[8:16]
    for choose_method in range(len(METHOD_CHOICE_now)):
        method_ATTACK = METHOD_CHOICE_now[choose_method]  # 选择攻击方法
        for test_model_i in test_model2attack:
            ORI_IMG_PATH = dataset_ATTACK  # 原始
            ADV_IMG_PATH = f"{RESULT_PATH}{method_ATTACK}/{MODEL_CHOICE[test_model_i]}/"  # 对抗
            test_robust(ORI_IMG_PATH, ADV_IMG_PATH, test_model_i, robust=ROBUST_MODE)

    choose_data = 2  # 前六个对应ImageNet/ProGAN
    dataset_ATTACK, save_ATTACK = DATASET_PATH[choose_data][0], DATASET_PATH[choose_data][1]
    test_model2attack = [3, 4, 5]
    METHOD_CHOICE_now = METHOD_CHOICE[16:]
    for choose_method in range(len(METHOD_CHOICE_now)):
        method_ATTACK = METHOD_CHOICE_now[choose_method]  # 选择攻击方法
        for test_model_i in test_model2attack:
            ORI_IMG_PATH = dataset_ATTACK  # 原始
            ADV_IMG_PATH = f"{RESULT_PATH}{method_ATTACK}/{MODEL_CHOICE[test_model_i]}/"  # 对抗
            test_robust(ORI_IMG_PATH, ADV_IMG_PATH, test_model_i, robust=ROBUST_MODE)

