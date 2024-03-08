import torch
from utils.train_dataset import TrainDataset
from utils.test_dataset import TestDataset 

from utils.utils import load_model_weights, load_fusion_net
from models.models import TextHeading, TextEncoder, ImageHeading
from models.fusion_nets import (LinearFusion, FCFM, CMF)
from models import iresnet, net 
from models import metrics
from models.network import NetworkBuilder
from utils.dataset_utils import *


###########   model   ############
def prepare_text_encoder(args):    
    text_encoder =  TextEncoder(args)
    text_encoder = torch.nn.DataParallel(text_encoder, device_ids=args.gpu_id).to(args.device)
    state_dict = torch.load(args.text_encoder_path)
    text_encoder.load_state_dict(state_dict['model'])

    text_head = TextHeading(args)
    text_head = torch.nn.DataParallel(text_head, device_ids=args.gpu_id).to(args.device)
    text_head.load_state_dict(state_dict['head'])
    del state_dict

    print("loading text encoder weights: ", args.text_encoder_path)
    return text_encoder, text_head


def prepare_image_head(args):
    print("loading image encoder: ", args.image_encoder_path)
    head = ImageHeading(args)

    head = torch.nn.DataParallel(head, device_ids=args.gpu_id).to(args.device)
    state_dict = torch.load(args.image_encoder_path)
    head.load_state_dict(state_dict['image_head'])
    return head 


def prepare_image_text_attr(args):
    print("loading image_text_attr: ", args.image_encoder_path)
    image_text_attr = metrics.Classifier(args.gl_text_dim, 40)

    image_text_attr = torch.nn.DataParallel(image_text_attr, device_ids=args.gpu_id).to(args.device)
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
    
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    print("loading pretrained arcface model: ", args.architecture)


    if train_mode == "my_own":
        state_dict = torch.load(args.backend_path)
        model.load_state_dict(state_dict['image_encoder'])
        print("my own arcface weight is loaded")

    if train_mode == "fixed" or train_mode == "my_own":
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        print("******* ArcFace weights are fixed **********")

    return model 



#### model for AdaFace 
def prepare_adaface(args, train_mode):
    device = args.device

    if args.architecture == "ir_18":
        weights_path = args.weights_adaface_18
    elif args.architecture == "ir_50":
        weights_path = args.weights_adaface_50

    model = net.build_model(args.architecture) 
    statedict = torch.load(weights_path)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    print("loading pretrained adaface model:  ", args.architecture)

    if train_mode == "my_own":
        state_dict = torch.load(args.backend_path)
        model.load_state_dict(state_dict['image_encoder'])
        print("my own adaface weight is loaded")

    if train_mode == "fixed" or train_mode == "my_own":
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        print("******* AdaFace weights are fixed **********")

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

