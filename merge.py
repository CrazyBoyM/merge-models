import os
import argparse
import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("model_0", type=str, help="Path to model 0")
parser.add_argument("model_1", type=str, help="Path to model 1")
parser.add_argument("--alpha", type=float, help="Alpha value, optional, defaults to 0.5", default=0.5, required=False)
parser.add_argument("--dump_path", type=str, default=None, help="Path to the output file.", required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--without_vae", action="store_true", help="Do not merge VAE", required=False)

args = parser.parse_args()

def load_weights(path, device):
  if path.endswith(".safetensors"):
      weights = load_file(path, device)
  else:
      weights = torch.load(path, device)
      weights = weights["state_dict"] if "state_dict" in weights else weights
  
  return weights

def save_weights(weights, path):
  if path.endswith(".safetensors"):
      save_file(weights, path)
  else:
      torch.save({"state_dict": weights}, path) 
        
device = args.device
theta_0 = load_weights(args.model_0, device=device)
theta_1 = load_weights(args.model_1, device=device)
alpha = args.alpha

for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
    # skip VAE model parameters to get better results(tested for anime models)
    # for anime model，with merging VAE model, the result will be worse (dark and blurry)
    if args.without_vae and "first_stage_model" in key:
        continue
        
    if "model" in key and key in theta_1:
        theta_0[key] = (1 - alpha) * theta_0[key] + alpha * theta_1[key]

for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
    if "model" in key and key not in theta_0:
        theta_0[key] = theta_1[key]

print("Saving...")

if args.dump_path is not None:
  save_weights(theta_0, args.dump_path)
else:
  weights0_name = os.path.basename(args.model_0)
  weights1_name = os.path.basename(args.model_1)
  merged_name = f"{weights0_name}_{weights1_name}.safetensors"
  dump_path = os.path.join(os.path.dirname(args.model_0), merged_name)
  save_weights(theta_0, dump_path)

print("Done!")
