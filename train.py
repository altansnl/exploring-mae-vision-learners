from maemodel import MAEPretainViT
import argparse


if __name__ == "__main__":

    # options for training
    parser = argparse.ArgumentParser()

    # default parameter setting for Vit-B
    parser.add_argument('--img_dim',  type=int, default=64, help='image dimensionality')
    parser.add_argument('--num_channels',  type=int, default=3, help='number of channels for the input image')
    parser.add_argument('--embed_dim',  type=int, default=768, help='encoder embedding dimensionality')
    parser.add_argument('--hidden_dim_ratio',  type=float, default=4., help='encoder hidden dimension ratio')   
    parser.add_argument('--num_heads',  type=int, default=12, help='encoder number of heads')
    parser.add_argument('--num_layers',  type=int, default=12, help='number of transformer layers in the encoder')
    parser.add_argument('--patch_size',  type=int, default=16, help='patch size')
    parser.add_argument('--decoder_embed_dim',  type=int, default=512, help='decoder embedding dimensionality')
    parser.add_argument('--decoder_hidden_dim_ratio',  type=float, default=4., help='encoder hidden dimension ratio') 
    parser.add_argument('--decoder_num_heads',  type=int, default=16, help='decoder number of heads')
    parser.add_argument('--decoder_num_layers',  type=int, default=8, help='number of layers in the decoder')
    parser.add_argument('--mask_ratio',  type=float, default=.75, help='mask ratio')

    opt = parser.parse_args()

    # initialize the MAE model
    mae = MAEPretainViT(
        img_dim=opt.img_dim,
        num_channels=opt.num_channels,
        enc_embed_dim=opt.embed_dim,
        enc_hidden_dim_ratio=opt.hidden_dim_ratio,
        enc_num_heads=opt.num_heads,
        enc_num_layers=opt.num_layers,
        patch_size=opt.patch_size,
        dec_embed_dim=opt.decoder_embed_dim,
        dec_hidden_dim_ratio=opt.decoder_hidden_dim_ratio,
        dec_num_heads=opt.decoder_num_heads,
        dec_num_layers=opt.decoder_num_layers,
        mask_ratio=opt.mask_ratio
    )
