import torch
from utils.train_dataset import TrainDataset
from utils.test_dataset import TestDataset 

from utils.utils import load_model_weights
from models.models import TextHeading, TextEncoder, ImageHeading
from models.fusion_nets import (LinearFusion, FCFM, CMF, CMF_FR)
from models import iresnet, net 
from models import metrics
from utils.dataset_utils import *

###########   model   ############
def prepare_text_encoder(args):    
    text_encoder =  TextEncoder(args).to(args.device)
    state_dict = torch.load(args.text_encoder_path)
    text_encoder.load_state_dict(state_dict['model'])

    text_head = TextHeading(args).to(args.device)
    text_head.load_state_dict(state_dict['head'])
    del state_dict

    print("loading text encoder weights: ", args.text_encoder_path)
    return text_encoder, text_head


def prepare_image_head(args):
    head = ImageHeading(args).to(args.device)
    state_dict = torch.load(args.image_encoder_path)
    head.load_state_dict(state_dict['image_head'])
    return head 


def prepare_image_text_attr(args):
    print("loading image_text_attr: ", args.image_encoder_path)
    image_text_attr = metrics.TopLayer(args.gl_text_dim, 40).to(args.device)
    state_dict = torch.load(args.image_encoder_path)
    image_text_attr.load_state_dict(state_dict['attr_head'])
    return image_text_attr 

    """
    unfreeze_layers = ['bn2', 'fc','features'] #'layer3.1', 'layer.4', 
    for name, param in model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
    """

### model for ArcFace
def prepare_arcface(args, train_mode):
    device = args.device
    if args.architecture == "ir_18":
        model = iresnet.iresnet18(pretrained=False, progress=True)
        weights_path = args.weights_arcface_18

    elif args.architecture == "ir_50":
        model = iresnet.iresnet50(pretrained=False, progress=True)
        weights_path = args.weights_arcface_50

    elif args.architecture == "ir_101":
        model = iresnet.iresnet101(pretrained=False, progress=True)
        weights_path = args.weights_arcface_101
    
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint)

    if train_mode == "my_own":
        state_dict = torch.load(args.backend_path)
        model.load_state_dict(state_dict['image_encoder'])
        for p in model.parameters():
            p.requires_grad = False

    elif train_mode == "fixed":
        for p in model.parameters():
            p.requires_grad = False 

    elif train_mode == "finetune":
        pass 

    model.to(device)
    model.eval()
    print("******* ArcFace weights are fixed **********")
    print("loading pretrained adaface model:  ", args.architecture)
    return model 



#### model for AdaFace 
def prepare_adaface(args, train_mode):
    device = args.device

    if args.architecture == "ir_18":
        weights_path = args.weights_adaface_18
        
    elif args.architecture == "ir_50":
        weights_path = args.weights_adaface_50

    elif args.architecture == "ir_101":
        weights_path = args.weights_adaface_101

    model = net.build_model(args.architecture) 
    statedict = torch.load(weights_path)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    

    if train_mode == "my_own":
        state_dict = torch.load(args.backend_path)
        model.load_state_dict(state_dict['image_encoder'])
        for p in model.parameters():
            p.requires_grad = False

    elif train_mode == "fixed":
        for p in model.parameters():
            p.requires_grad = False 

    elif train_mode == "finetune":
        pass 
    
    model.to(device)
    model.eval()
    print("******* AdaFace weights are fixed **********")
    print("loading pretrained adaface model:  ", args.architecture)
    return model 



#### model for AdaFace 
def prepare_magface(args, train_mode):
    device = args.device

    if args.architecture == "ir_50":
        model = iresnet.iresnet50(pretrained=False, progress=True)
        weights_path = args.weights_magface_50
        mag_dict = torch.load(weights_path)
        del mag_dict["state_dict"]["parallel_fc.weight"]

    elif args.architecture == "ir_101":
        model = iresnet.iresnet101(pretrained=False, progress=True)
        weights_path = args.weights_magface_101
        mag_dict = torch.load(weights_path)
        del mag_dict["state_dict"]["fc.weight"]

    
    state_d = {}
    for k, v in mag_dict["state_dict"].items():
        state_d[k[16:]] = v

    del mag_dict
    
    model.load_state_dict(state_d)
    model.to(device)
    #model = torch.nn.DataParallel(model, device_ids=args.gpu_id).to(device)
    print("loading pretrained magface model:  ", args.architecture)

    if train_mode == "my_own":
        state_dict = torch.load(args.backend_path)
        model.load_state_dict(state_dict['image_encoder'])
        print("my own magface weight is loaded")

    if train_mode == "fixed" or train_mode == "my_own":
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        print("******* Magface weights are fixed **********")

    return model 


def prepare_fusion_net(args):
    if args.fusion_type == "linear":
        fusion_net = LinearFusion(args).to(args.device)

    elif args.fusion_type == "fcfm":
        fusion_net = FCFM(channel_dim = args.gl_img_dim).to(args.device)

    elif args.fusion_type == "CMF":
        fusion_net = CMF(args).to(args.device)

    elif args.fusion_type == "CMF_FR":
        fusion_net = CMF_FR(args).to(args.device)

    checkpoint = torch.load(args.image_encoder_path)
    fusion_net.load_state_dict(checkpoint["net"])
    return fusion_net 




############## dataloader #############
def prepare_train_val_loader(args):
        
    train_filenames, train_captions, train_att_masks, train_attr_label,\
    valid_filenames, valid_captions, valid_att_masks, valid_attr_label,\
    test_filenames, test_captions, test_att_masks, test_attr_label =  load_text_data_Bert(args.data_dir, args)

    train_ds = TrainDataset(train_filenames, 
                            train_captions, 
                            train_att_masks,
                            train_attr_label, 
                            split="train", 
                            args=args)

    valid_ds = TestDataset(valid_filenames, 
                            valid_captions, 
                            valid_att_masks, 
                            split="valid", 
                            args=args)

    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        drop_last=True,
        num_workers=args.num_workers, 
        shuffle=True)

    valid_dl = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size= args.batch_size * 8, 
        drop_last=False,
        num_workers= args.num_workers, 
        shuffle=False)

    return train_dl, valid_dl


def prepare_test_loader(args):
    train_filenames, train_captions, train_att_masks, train_attr_label,\
    valid_filenames, valid_captions, valid_att_masks, valid_attr_label,\
    test_filenames, test_captions, test_att_masks, test_att_label =  load_text_data_Bert(args.data_dir, args)
    
    test_ds =  TestDataset(valid_filenames, 
                            valid_captions, 
                            valid_att_masks,
                            split="valid", 
                            args=args)

    test_dl = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        drop_last = False, 
        shuffle=False)

    return test_dl