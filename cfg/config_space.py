from types import SimpleNamespace
import torch 

# Face2Text dataset
face2text_cfg = SimpleNamespace(
    data_dir = "./data/face2text",  
    test_ver_acc_list= "./data/face2text/images/test_ver_acc.txt",
    test_ver_list= "./data/face2text/images/test_ver.txt", 
    valid_ver_list= "./data/face2text/images/valid_ver.txt",
    num_classes = 4500,
    bert_words_num = 40, 
    captions_per_image = 4,
    test_sub = 1193 
)

# CelebA dataset
celeba_cfg = SimpleNamespace(
    data_dir= "./data/celeba",  
    test_ver_acc_list= "./data/celeba/images/test_ver_acc.txt",
    test_ver_list= "./data/celeba/images/test_ver.txt", 
    valid_ver_list= "./data/celeba/images/valid_ver.txt",
    num_classes= 4500, 
    bert_words_num = 32,
    captions_per_image= 10,
    test_sub = 1217
)


# CelebA-Dialog dataset
celeba_dialog_cfg = SimpleNamespace(
    data_dir= "./data/celeba_dialog",  
    test_ver_acc_list= "./data/celeba_dialog/images/test_ver_acc.txt",
    test_ver_list= "./data/celeba_dialog/images/test_ver.txt", 
    valid_ver_list= "./data/celeba_dialog/images/valid_ver.txt",
    num_classes= 8000, 
    bert_words_num = 32,
    captions_per_image = 1,
    test_sub = 1677
)

# CelebA-Dialog dataset
LFW_cfg = SimpleNamespace(
    data_dir= "./data/LFW",  
    test_ver_list= "./data/LFW/images/test_pairs.txt", 
    valid_ver_list= "./data/LFW/images/test_pairs.txt",
    num_classes= 1468, 
    bert_words_num = 32,
    captions_per_image = 1,
    test_sub = 4281
)

# CelebA-Dialog dataset
CALFW_cfg = SimpleNamespace(
    data_dir= "./data/CALFW",  
    test_ver_list= "./data/CALFW/images/test_pairs.txt", 
    valid_ver_list= "./data/CALFW/images/test_pairs.txt",
    num_classes= 1468, 
    bert_words_num = 32,
    captions_per_image = 1,
    test_sub = 4281
)

# CelebA-Dialog dataset
AGEDB_cfg = SimpleNamespace(
    data_dir= "./data/AGEDB",  
    test_ver_list= "./data/AGEDB/images/test_pairs.txt", 
    valid_ver_list= "./data/AGEDB/images/test_pairs.txt",
    num_classes= 1468, 
    bert_words_num = 32,
    captions_per_image = 1,
    test_sub = 4281
)

setup_cfg = SimpleNamespace(
    weights_adaface_18 = "./weights/pretrained/adaface_ir18_webface4m.ckpt",
    weights_adaface_50 = "./weights/pretrained/adaface_ir50_ms1mv2.ckpt",
    weights_adaface_101 = "./weights/pretrained/adaface_ir101_ms1mv2.ckpt",

    weights_arcface_18 = "./weights/pretrained/arcface_ir18_ms1mv3.pth", 
    weights_arcface_50 = "./weights/pretrained/arcface_ir50_ms1mv3.pth", 
    weights_arcface_101 ="./weights/pretrained/arcface_ir101_ms1mv3.pth", 

    weights_magface_50 = "./weights/pretrained/magface_ir50_ms1mv2.pth", 
    weights_magface_101 ="./weights/pretrained/magface_ir101_ms1mv2.pth", 

    metric= "arc_margin", 
    easy_margin= False,
    loss= "focal_loss", 
    use_se= False,

    bert_config=  "bert-base-uncased", #distilbert-base-uncased 
    align_config= "kakaobrain/align-base",
    clip_config= "openai/clip-vit-base-patch32",
    blip_config= "Salesforce/blip-image-captioning-base",
    falva_config= "facebook/flava-full",

    # machine setup
    num_workers= 4, 
    gpu_id= [1], #1
    device = torch.device("cuda:1"),
    manual_seed= 100
)