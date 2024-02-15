import torch
from utils.train_dataset import TrainDataset
from utils.test_dataset import TestDataset 

from utils.utils import load_model_weights, load_fusion_net
from models.models import TextHeading, TextEncoder, ImageHeading
from models.fusion_nets import (LinearFusion, FCFM, CMF)
from models import iresnet, net 
from models.network import NetworkBuilder
from utils.dataset_utils import *


###########   model   ############
def prepare_text_encoder(args):    
    text_encoder =  TextEncoder(args)
    text_encoder = torch.nn.DataParallel(text_encoder, device_ids=args.gpu_id).cuda()
    state_dict = torch.load(args.text_encoder_path)
    text_encoder.load_state_dict(state_dict['model'])

    text_head = TextHeading(args)
    text_head = torch.nn.DataParallel(text_head, device_ids=args.gpu_id).cuda()
    text_head.load_state_dict(state_dict['head'])
    del state_dict

    print("loading text encoder weights: ", args.text_encoder_path)
    return text_encoder, text_head


def prepare_image_head(args):
    print("loading image encoder: ", args.image_encoder_path)
    head = ImageHeading(args)

    head = torch.nn.DataParallel(head, device_ids=args.gpu_id).cuda()
    state_dict = torch.load(args.image_encoder_path)
    head.load_state_dict(state_dict['image_head'])
    return head 


### model for ArcFace
def prepare_arcface(args):
    device = args.device
    model = iresnet.iresnet18(pretrained=False, progress=True)

    checkpoint = torch.load(args.weights_arcface)
    model.load_state_dict(checkpoint)

    model = torch.nn.DataParallel(model, device_ids=args.gpu_id).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print("loading pretrained arcface model")
    return model 


#### model for AdaFace 
def prepare_adaface(args):
    device = args.device
    architecture = "ir_18"

    model = net.build_model(architecture)    
    statedict = torch.load(args.weights_adaface)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print("loading pretrained adaface model")
    return model 


#### model for MagFace 
def prepare_magface(args):
    device = args.device
    resnet = NetworkBuilder(arch = "iresnet18")
    resnet = torch.nn.DataParallel(resnet, device_ids=args.gpu_id).to(device)

    mag_dict = torch.load(args.weights_magface)['state_dict']
    del mag_dict["module.fc.weight"]
    resnet.load_state_dict(mag_dict)
    
    for p in resnet.parameters():
        p.requires_grad = False
    resnet.eval()
    print("loading pretrained magface model")
    return resnet 


def prepare_fusion_net(args):
    # fusion models
    if args.fusion_type == "linear":
        net = LinearFusion(args)

    elif args.fusion_type == "fcfm":
        net = FCFM(channel_dim = args.gl_img_dim)

    elif args.fusion_type == "CMF":
        net = CMF(args)

    net = torch.nn.DataParallel(net, device_ids=args.gpu_id).to(args.device)
    print("loading checkpoint; epoch: ", args.image_encoder_path)
    net = load_fusion_net(net, args.image_encoder_path) 
    
    return net


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
    
    test_ds =  TestDataset(test_filenames, 
                            test_captions, 
                            test_att_masks,
                            split="test", 
                            args=args)

    test_dl = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        drop_last = False, 
        shuffle=False)

    return test_dl

