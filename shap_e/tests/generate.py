import os
import time
import torch
import glob
import imageio
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import requests.exceptions

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import (
    create_pan_cameras,
    decode_latent_images,
    decode_latent_mesh,
)

def robust_load_model(model_name, device, max_attempts=5, delay=10):
    """Attempts to load a model with retry logic for network failures."""
    for attempt in range(1, max_attempts + 1):
        try:
            model = load_model(model_name, device=device)
            print(f"[{model_name}] Successfully loaded on {device} (Attempt {attempt}).")
            return model
        except requests.exceptions.RequestException as e:
            print(f"[{model_name}] Network error (Attempt {attempt}/{max_attempts}): {e}")
        except Exception as e:
            print(f"[{model_name}] Error loading model (Attempt {attempt}/{max_attempts}): {e}")
        if attempt < max_attempts:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    raise RuntimeError(f"Failed to load model {model_name} after {max_attempts} attempts.")

def init_process(rank, world_size, batch_size, fn, backend="nccl"):
    """Initialize the process group for distributed processing."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, world_size, batch_size)
    dist.destroy_process_group()

def get_next_available_filename(output_dir, base_name, extension):
    """Find the next available unique filename without overwriting."""
    existing_files = glob.glob(os.path.join(output_dir, f"{base_name}_*.{extension}"))
    existing_indices = sorted(
        [int(f.split("_")[-1].split(".")[0]) for f in existing_files if f.split("_")[-1].split(".")[0].isdigit()]
    )

    index = (existing_indices[-1] + 1) if existing_indices else 0

    return os.path.join(output_dir, f"{base_name}_{index}.{extension}")

def train(rank, world_size, batch_size):
    """Distributed processing for latent generation, broadcasting, and 3D model decoding."""
    print(f"[Rank {rank}] Process started.")
    torch.cuda.set_device(rank)

    output_dir = os.path.abspath("generated_3d_models")
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    xm = robust_load_model("transmitter", device=torch.device("cuda", rank))
    xm = DDP(xm, device_ids=[rank], output_device=rank)

    local_batch_size = batch_size // world_size
    extra = batch_size % world_size

    if rank < extra:
        local_batch_size += 1

    latents = None
    if local_batch_size > 0:
        print(f"[Rank {rank}] Generating {local_batch_size} latents...")
        model = robust_load_model("text300M", device=torch.device("cuda", rank))
        diffusion = diffusion_from_config(load_config("diffusion"))

        prompt = os.getenv("PROMPT", "a man")
        latents = sample_latents(
            batch_size=local_batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=7.5,
            model_kwargs=dict(texts=[prompt] * local_batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=torch.cuda.is_available(),
            use_karras=True,
            karras_steps=32,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        print(f"[Rank {rank}] Latents generated: {len(latents)}")

    if latents is not None:
        latents = [torch.tensor(lt, dtype=torch.float32, device="cuda").clone().detach() for lt in latents]
        latents_tensor = torch.stack(latents, dim=0)
    else:
        latents_tensor = torch.zeros((batch_size, 1048576), dtype=torch.float32, device="cuda")

    all_latents = [torch.zeros_like(latents_tensor) for _ in range(world_size)]
    dist.all_gather(all_latents, latents_tensor)

    latents = [lt for lt in all_latents if lt.sum().item() != 0]

    for i, latent in enumerate(latents):
        if (i // world_size) % world_size == rank:
            ply_filename = get_next_available_filename(output_dir, "object", "ply")
            obj_filename = get_next_available_filename(output_dir, "object", "obj")
            gif_filename = get_next_available_filename(output_dir, "object", "gif")

            print(f"[Rank {rank}] Processing latent {i}:")
            mesh = decode_latent_mesh(xm.module, latent)
            tri_mesh = mesh.tri_mesh()
            with open(ply_filename, "wb") as f:
                tri_mesh.write_ply(f)
            with open(obj_filename, "w") as f:
                tri_mesh.write_obj(f)
            print(f"[Rank {rank}] Mesh files saved for latent {i}.")

            # **Generate GIF Preview**
            try:
                cameras = create_pan_cameras(128, torch.device("cuda", rank))
                images = decode_latent_images(xm.module, latent, cameras, rendering_mode="nerf")
                imageio.mimsave(gif_filename, images, duration=0.08, loop=0)
                print(f"[Rank {rank}] GIF saved for latent {i}.")
            except Exception as e:
                print(f"[Rank {rank}] GIF generation error for latent {i}: {e}")

    print(f"[Rank {rank}] Decoding complete.")
    dist.barrier()

def main():
    world_size = torch.cuda.device_count()
    batch_size = int(os.getenv("BATCH_SIZE", 1))
    used_gpus = min(world_size, batch_size)
    mp.spawn(init_process, args=(used_gpus, batch_size, train), nprocs=used_gpus, join=True)

if __name__ == "__main__":
    main()