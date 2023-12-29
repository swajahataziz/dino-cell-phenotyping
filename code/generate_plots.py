import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
import umap
import os
import seaborn as sns
import yaml
import glob
import argparse
import utils
from pathlib import Path

#translate channel indexes to channel names
def get_channel_name_combi(channel_number, channel_dict):
    name_of_channel_combi = ""
    #for channel_number in iter(str(channel_combi_num)):
    name_of_channel_combi = "_".join([name_of_channel_combi, channel_dict[int(channel_number)]])
    print(f'Channel dict {channel_dict}')
    print(f'Name of channel combi {name_of_channel_combi}')

    return name_of_channel_combi

def get_channel_number_combi(channel_names, channel_dict):
    channel_combi = ""
    for channel_name in channel_names.split('_'):
        for key, value in channel_dict.items():
            if value == channel_name:
                channel_combi = "".join([channel_combi, str(key)])
    return channel_combi

def get_channel_name_combi_list(selected_channels, channel_dict):
    channel_names = []
    for channel_combi in selected_channels:
        channel_names.append(get_channel_name_combi(channel_combi,channel_dict))
    return channel_names

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'], #\
                # + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")

    #save settings
    parser.add_argument('--output_dir', default='/fsx/data', type=str)
    parser.add_argument('--full_ViT_name', default='ViT_base_p16', type=str, help='name channel combi ViT')
    parser.add_argument('--name_of_run', default='/recent_run', type=str)
    parser.add_argument('--selected_channels', default=[0], nargs='+', type=int, help="""list of channel indexes of the .tiff images which should be used to create the tensors.""")
    parser.add_argument('--channel_dict', default="Brightfield", type=str,help="""name of the channels in format as dict channel_number, channel_name.""")
    parser.add_argument('--n_neighbors', default="15", type=int,help="""num neighbors for UMAP eval""")
    parser.add_argument('--min_dist', default="0.1", type=float,help="""Min distance for UMAP eval""")
    parser.add_argument('--metric', default="euclidean", type=str,help="""Name of metric for UMAP eval.""")
    parser.add_argument('--spread', default="1.1", type=float,help="""Spread for UMAP eval.""")
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--topometry_plots', default=False, type=utils.bool_flag, help='Whether to create topometry plots (Default: False).')

    return parser


#UMAP
def fit_umap(data, n_neighbors, min_dist, metric, spread, epochs):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, spread=spread, n_epochs=epochs, random_state=42)
    umap_embedding = umap_model.fit_transform(data)
    return umap_embedding


def make_plot(embedding, labels, save_dir, file_name,name="Emb type", description="details"):
    sns_plot = sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, s=14, palette=custom_palette, linewidth=0, alpha=0.9)
    plt.suptitle(f"{name}_{file_name}", fontsize=8)
    sns_plot.tick_params(labelbottom=False)
    sns_plot.tick_params(labelleft=False)
    sns_plot.tick_params(bottom=False)
    sns_plot.tick_params(left=False)
    sns_plot.set_title("CLS Token embedding of "+str(len(labels))+" cells with a dimensionality of "+str(features.shape[1])+" \n"+description, fontsize=6)
    sns.move_legend(sns_plot, "lower left", title='Classes', prop={'size': 5}, title_fontsize=6, markerscale=0.5)
    sns.set(rc={"figure.figsize":(14, 10)})
    sns.despine(bottom = True, left = True)
    sns_plot.figure.savefig(f"{save_dir}{file_name}{name}.png", dpi=325)
    sns_plot.figure.savefig(f"{save_dir}pdf_format/{file_name}{name}.pdf")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    if len(args.selected_channels) > 1:
        args.selected_channels = list(map(int, args.selected_channels[0].split(',')))
    else:
        args.selected_channels = args.selected_channels


    Path(f"{args.output_dir}/pdf_format").mkdir(parents=True, exist_ok=True)
    
    #os.makedirs(f"{save_dir}pdf_format", exist_ok=True)
        
    name_of_run = args.name_of_run
    sk_save_dir = args.output_dir
    save_dir_downstream_run = sk_save_dir+"/"+name_of_run
    
    
    #channel_aTub_model_checkpoint2_features.csv
    features_path = f"{save_dir_downstream_run}/CLS_features/"
    
    features_file = glob.glob(features_path+'*_features.csv')[0]
    
    labels_file = f"{save_dir_downstream_run}/CLS_features/class_labels.csv"
    
    
    #load data
    features = np.genfromtxt(features_file, delimiter = ',')
    class_labels_pd = pd.read_csv(labels_file, header=None)
    class_labels = class_labels_pd[0].tolist()
        
    dino_vit_name = args.full_ViT_name
    
    selected_channel_str = "".join(str(x) for x in args.selected_channels)
    channel_dict = dict(zip(args.selected_channels, args.channel_dict.split(',')))
    print(channel_dict)
    print(selected_channel_str)
        
    channel_names = get_channel_name_combi_list(args.selected_channels, channel_dict)

    print(f'Channel names: {channel_names}')
    
    save_dir = f"{save_dir_downstream_run}/{dino_vit_name}_channel{channel_names[0]}_analyses/embedding_plots/"
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True) 
    Path(f"{save_dir}pdf_format").mkdir(parents=True, exist_ok=True)
    
    file_name = f"{channel_names[0]}_{dino_vit_name}"
    
    n_neighbors = args.n_neighbors
    min_dist = args.min_dist

    metric = args.metric
    spread = args.spread
    epochs = args.epochs
    
    umap_embedding = fit_umap(features, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, spread=spread, epochs=epochs)
    
    custom_palette = sns.color_palette("hls", len(set(class_labels)))
    
    
    make_plot(umap_embedding, class_labels, save_dir=save_dir, file_name=file_name, name="umap",description=f"n_neighbors:{n_neighbors}, min_dist={min_dist}, metric={metric}, spread={spread}, epochs={epochs}")
    
    ########################### Additional plots from https://topometry.readthedocs.io/en/latest/ ###########################
    if args.topometry_plots:
    
        import topo as tp
    
        os.makedirs(f"{save_dir}/topometry_plots", exist_ok=True)
        os.makedirs(f"{save_dir}/topometry_plots/pdf_format", exist_ok=True)
    
        save_dir_topo = f"{save_dir}/topometry_plots/"
        # Learn topological metrics and basis from data. The default is to use diffusion harmonics.
        tg = tp.TopOGraph()
    
        print('running all combinations')
        tg.run_layouts(features, n_components=2,
                            bases=['diffusion', 'fuzzy'],
                            graphs=['diff', 'fuzzy'],
                            layouts=['tSNE', 'MAP', 'MDE', 'PaCMAP', 'TriMAP', 'NCVis'])
    
        make_plot(tg.db_diff_MAP, class_labels, name="db_diff_MAP", save_dir=save_dir_topo)
        make_plot(tg.db_fuzzy_MAP, class_labels, name="db_fuzzy_MAP", save_dir=save_dir_topo)
        make_plot(tg.db_diff_MDE, class_labels, name="db_diff_MDE", save_dir=save_dir_topo)
        make_plot(tg.db_fuzzy_MDE, class_labels, name="db_fuzzy_MDE", save_dir=save_dir_topo)
        make_plot(tg.db_PaCMAP, class_labels, name="db_PaCMAP", save_dir=save_dir_topo)
        make_plot(tg.db_TriMAP, class_labels, name="db_TriMAP", save_dir=save_dir_topo)
        make_plot(tg.db_tSNE, class_labels, name="db_tSNE", save_dir=save_dir_topo)
        make_plot(tg.fb_diff_MAP, class_labels, name="fb_diff_MAP", save_dir=save_dir_topo)
        make_plot(tg.fb_fuzzy_MAP, class_labels, name="fb_fuzzy_MAP", save_dir=save_dir_topo)
        make_plot(tg.fb_diff_MDE, class_labels, name="fb_diff_MDE", save_dir=save_dir_topo)
        make_plot(tg.fb_fuzzy_MDE, class_labels, name="fb_fuzzy_MDE", save_dir=save_dir_topo)
        make_plot(tg.fb_PaCMAP, class_labels, name="fb_PaCMAP", save_dir=save_dir_topo)
        make_plot(tg.fb_TriMAP, class_labels, name="fb_TriMAP", save_dir=save_dir_topo)
        make_plot(tg.fb_tSNE, class_labels, name="fb_tSNE", save_dir=save_dir_topo)