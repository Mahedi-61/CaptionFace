from types import SimpleNamespace
import torch 

# Face2Text dataset
face2text_cfg = SimpleNamespace(
    data_dir = "./data/face2text",  
    test_pair_list = "./data/face2text/images/test_199_sub.txt", 
    valid_pair_list = "./data/face2text/images/valid_199_sub.txt",
    num_classes = 4500,
    bert_words_num = 40, 
    captions_per_image = 4,
    test_sub = 1193 
)

# CelebA dataset
celeba_cfg = SimpleNamespace(
    data_dir= "./data/celeba",  
    test_pair_list= "./data/celeba/images/test_199_sub.txt", 
    valid_pair_list= "./data/celeba/images/valid_199_sub.txt",
    num_classes= 4500, 
    bert_words_num = 32,
    captions_per_image= 10,
    test_sub = 1217
)

# CelebA-Dialog dataset
celeba_dialog_cfg = SimpleNamespace(
    data_dir= "./data/celeba_dialog",  
    test_pair_list= "./data/celeba_dialog/images/test_199_sub.txt", 
    valid_pair_list= "./data/celeba_dialog/images/valid_199_sub.txt",
    num_classes= 8000, 
    bert_words_num = 32,
    captions_per_image = 1,
    test_sub = 1677
)

setup_cfg = SimpleNamespace(
    weights_adaface_18 = "./weights/pretrained/adaface_ir18_webface4m.ckpt",
    weights_adaface_50 = "./weights/pretrained/adaface_ir50_ms1mv2.ckpt",

    weights_arcface_18= "./weights/pretrained/arcface_ir18_ms1mv3.pth", 
    weights_arcface_50= "./weights/pretrained/arcface_ir50_ms1mv3.pth", 


    weights_magface= "./weights/pretrained/magface_iresnet18_casia_dp.pth",
    metric= "arc_margin", 
    easy_margin= False,
    loss= "focal_loss", 
    use_se= False,

    bert_config=  "bert-base-uncased", #distilbert-base-uncased 
    align_config= "kakaobrain/align-base",
    clip_config= "openai/clip-vit-base-patch32",
    blip_config= "Salesforce/blip-image-captioning-base",
    falva_config= "facebook/flava-full",
    groupvit_config= "nvidia/groupvit-gcc-yfcc",

    # machine setup
    num_workers= 4, 
    gpu_id= [0], #1
    device = torch.device("cuda:0"),
    manual_seed= 100
)