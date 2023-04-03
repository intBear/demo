import argparse
import getpass
import imageio.v2 as imageio
import json
import os
import random
import torch
import util
from siren import Encoder, Siren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer
import wandb

parser = argparse.ArgumentParser()
# Basic arguments
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=50000)
parser.add_argument("-fm", "--feature_dim", help="feature_dimension", type=int, default=8)
parser.add_argument("-bd", "--base_lod", help="base level of detail", type=int, default=5)
parser.add_argument("-nds", "--num_lods", help="number of lods", type=int, default=4)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=1e-3)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=64)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=2)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)

# Source arguments
parser.add_argument("-ip", "--input_type", help="Path to save logs", default="2D")
parser.add_argument("-sp", "--sample_type", help="Path to save logs", default="cat")
parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", type=int, default=0)
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=1)

# ChannelCoder arguments
parser.add_argument("-cd", "--tcn", help="dimension of channel input ", type=int, default=64)
parser.add_argument("-cp", "--chan_type", help="channel type", default="awgn")
parser.add_argument("-cm", "--chan_param", help="channel parameter(SNR)", type=int, default=13)
parser.add_argument("-pa", "--pass_channel", help="whether to pass channel", type=int, default=0)

# wandb arguments
parser.add_argument("--use_wandb", type=int, default=0)
parser.add_argument("--wandb_project_name", type=str, default="Hybrid")
parser.add_argument("--wandb_entity", type=str, default="intbear")
parser.add_argument("--wandb_job_type", help="Wandb job type. This is useful for grouping runs together.", type=str, default=None)

args = parser.parse_args()

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.use_wandb:
    # Initialize wandb experiment
    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        job_type=args.wandb_job_type,
        config=args,
    )

if args.full_dataset:
    min_id, max_id = 1, 24  # Kodak dataset runs from kodim01.png to kodim24.png
else:
    min_id, max_id = args.image_id, args.image_id

# Dictionary to register mean values (both full precision and half precision)
# results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

# Create directory to store experiments
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

# Fit images
for i in range(min_id, max_id + 1):
    print(f'Image {i}')

    # Load image
    img = imageio.imread(f"kodak-dataset/kodim{str(i).zfill(2)}.png")
    img = transforms.ToTensor()(img).float().to(device, dtype)
    img = img.permute(0, 1, 2).unsqueeze(0)
    # Setup model
    func_rep = Encoder(args).to(device)

    # Set up training
    trainer = Trainer(func_rep, lr=args.learning_rate)
    coordinates = util.image_to_coordinates(img)
    coordinates, img = coordinates.to(device, dtype), img.to(device, dtype)

    # Calculate model size. Divide by 8000 to go from bits to kB
    # model_size = util.model_size_in_bits(func_rep) / 8000.
    # print(f'Model size: {model_size:.1f}kB')
    # fp_bpp = util.bpp(model=func_rep, image=img)
    # print(f'Full precision bpp: {fp_bpp:.2f}')
    #
    # Train model in full precision
    trainer.train(coordinates, img, args, num_iters=args.num_iters)
    # print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
    #
    # # Log full precision results
    # results['fp_bpp'].append(fp_bpp)
    # results['fp_psnr'].append(trainer.best_vals['psnr'])
    #
    # # Save best model
    torch.save(trainer.best_model, args.logdir + f'/best_model_{i}.pt')

    # Update current model to be best model
    func_rep.load_state_dict(trainer.best_model)

    # Save full precision image reconstruction
    with torch.no_grad():
        img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
        save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/fp_reconstruction_{i}.png')

#     # Convert model and coordinates to half precision. Note that half precision
#     # torch.sin is only implemented on GPU, so must use cuda
#     if torch.cuda.is_available():
#         func_rep = func_rep.half().to('cuda')
#         coordinates = coordinates.half().to('cuda')
#
#         # Calculate model size in half precision, which can shorten the inference time
#         hp_bpp = util.bpp(model=func_rep, image=img)
#         results['hp_bpp'].append(hp_bpp)
#         print(f'Half precision bpp: {hp_bpp:.2f}')
#
#         # Compute image reconstruction and PSNR
#         with torch.no_grad():
#             img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1).float()
#             hp_psnr = util.get_clamped_psnr(img_recon, img)
#             save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/hp_reconstruction_{i}.png')
#             print(f'Half precision psnr: {hp_psnr:.2f}')
#             results['hp_psnr'].append(hp_psnr)
#     else:
#         results['hp_bpp'].append(fp_bpp)
#         results['hp_psnr'].append(0.0)
#
#     # Save logs for individual image
#     with open(args.logdir + f'/logs{i}.json', 'w') as f:
#         json.dump(trainer.logs, f)
#
#     print('\n')
#
# print('Full results:')
# print(results)
# with open(args.logdir + f'/results.json', 'w') as f:
#     json.dump(results, f)
#
# # Compute and save aggregated results
# results_mean = {key: util.mean(results[key]) for key in results}
# with open(args.logdir + f'/results_mean.json', 'w') as f:
#     json.dump(results_mean, f)

# print('Aggregate results:')
# print(f'Full precision, bpp: {results_mean["fp_bpp"]:.2f}, psnr: {results_mean["fp_psnr"]:.2f}')
# print(f'Half precision, bpp: {results_mean["hp_bpp"]:.2f}, psnr: {results_mean["hp_psnr"]:.2f}')
