import argparse, sys 
from fc_train import Trainer
from fc_test import Tester
import pprint

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',          dest="train",       help='train the GPTFace model',   action='store_true')
    parser.add_argument('--test',           dest="train",       help='evaluate the GPTFace model',action='store_false')

    parser.add_argument('--attr_loss',      dest="is_attr",     help='Employing attribute loss', action='store_true')
    parser.add_argument('--generate',       dest="gen_text",    help='generates text during testing', action='store_true')
    parser.add_argument('--sa',             dest="is_sa",       help='Employing self attention in PN', action='store_true')

    parser.set_defaults(train=False)
    parser.set_defaults(attr_loss=False)
    parser.set_defaults(gen_text=False)
    parser.set_defaults(sa=False)

    parser.add_argument('--dataset',            type=str,   default="celeba",    help='Name of the datasets: celeba|LFW |CALFW | AGEDB')
    parser.add_argument('--arch',               type=str,   default="arcface", help='Image encoder arcface|adaface')
    parser.add_argument('--batch_size',         type=int,   default=64,         help='Batch size')
    parser.add_argument('--epochs',             type=int,   default=8,          help='Number of epochs')
    parser.add_argument('--saved_model_file',   type=str,   default="model_celeba_arcface_18_w_attr_full.pth", help='The GPTFace model to load for test')

    parser.add_argument('--lr',                 type=float, default=1e-4,   help='Learning rate')
    parser.add_argument('--freeze_epochs',      type=int,   default=3,      help='Epochs on of freezing GPT layers')

    parser.add_argument('--resnet_layer',       type=int,   default=18,     help='Number of ResNet layers 18|50|101')
    parser.add_argument('--mlp_dropout',        type=float, default=0.1,    help='Dropout probability for MLP')
    parser.add_argument('--emb_dropout',        type=float, default=0.1,    help='Dropout probability for Embedding layer')

    parser.add_argument('--residual_dropout',   type=float, default=0.1,    help='Dropout rate in residual connection')
    parser.add_argument('--attention_dropout',  type=float, default=0.1,    help='Dropout rate in attention')
    parser.add_argument('--depth',              type=int,   default=12,     help='Depth of the GPT layers')
    parser.add_argument('--seq_len',            type=int,   default=1024,   help='length of the sequence')

    parser.add_argument('--num_heads',          type=int,   default=12,     help='Number of heads')
    parser.add_argument('--mlp_ratio',          type=int,   default=4,      help='Ratio in squeeze in GPT MLP block')

    parser.add_argument('--model_path',         type=str,   default="./weights/face_caption", help='model directory')
    parser.add_argument('--weights_path',       type=str,   default="./weights/pretrained", help='pretrained weights directory')

    parser.add_argument('--weight_adaface_18',  type=str,   default="adaface_ir18_webface4m.ckpt", help='Adaface (iResNet18 backend)')
    parser.add_argument('--weight_arcface_18',  type=str,   default="arcface_ir18_ms1mv3.pth",     help='ArcFace (iResNet18 backend)')

    parser.add_argument('--weight_adaface_50',  type=str,   default="adaface_ir50_ms1mv2.ckpt", help='AdaFace (iResNet50 backend)')
    parser.add_argument('--weight_arcface_50',  type=str,   default="arcface_ir50_ms1mv3.pth",  help='ArcFace (iResNet50 backend)')
    
    parser.add_argument('--device',             type=str,   default='cuda', help='Hardware accelerator type')
    parser.add_argument('--embed_dim',          type=int,   default=768,    help='Embedding dimension')
    parser.add_argument('--vocab_size',         type=int,   default=50_257, help='Vocabulary size')
   
    return  parser.parse_args(argv)


if __name__ == "__main__": 
    config = parse_arguments(sys.argv[1:])
    pprint.pp(config)

    if config.train == True:
        Trainer(config).train()

    elif config.train == False:
        Tester(config).test()

"""
RUN THE CODE
python3 src/gptface_main.py --test --generate --dataset AGEDB --arch arcface
"""