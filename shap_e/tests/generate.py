import os
import time
import torch
import glob
import imageio
import torch.distributed as dist
import torch.multiprocessing as mp
import shutil
import subprocess

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import (
    create_pan_cameras,
    decode_latent_images,
    decode_latent_mesh,
)

def robust_load_model(model_name, device, max_attempts=5, delay=10):
    for attempt in range(1, max_attempts + 1):
        try:
            model = load_model(model_name, device=device)
            print(f"[{model_name}] Successfully loaded on {device} (Attempt {attempt}).")
            return model
        except Exception as e:
            print(f"[{model_name}] Error loading model (Attempt {attempt}/{max_attempts}): {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to load model {model_name} after {max_attempts} attempts.")

def generate_gif(xm, latent, filename, device):
    try:
        cameras = create_pan_cameras(128, device)
        images = decode_latent_images(xm.module, latent, cameras, rendering_mode="nerf")

        if not images or len(images) == 0:
            print(f"Skipping empty GIF generation: {filename}")
            return

        imageio.mimsave(filename, images, duration=0.08, loop=0)
        print(f"GIF saved: {filename}")
    except Exception as e:
        print(f"GIF generation error for {filename}: {e}")

def get_next_filename(output_dir, extension):
    existing_files = glob.glob(os.path.join(output_dir, f"object_*{extension}"))
    numbers = [int(f.split("_")[-1].split(".")[0]) for f in existing_files if f.split("_")[-1].split(".")[0].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1
    return os.path.join(output_dir, f"object_{next_number}{extension}")

def clean_and_thicken_mesh_in_blender(obj_filename, ply_filename):
    blender_path = os.path.expanduser("~/software/blender/blender")
    subprocess.run([blender_path, "--background", "--python-expr",
        f"""
import bpy
import sys

obj_path, ply_path = sys.argv[-2], sys.argv[-1]

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath=obj_path, use_split_groups=False)
mesh = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = mesh
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.01)
bpy.ops.mesh.fill_holes(sides=6)
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

solidify_mod = mesh.modifiers.new(name='Solidify', type='SOLIDIFY')
solidify_mod.thickness = 0.05
bpy.ops.object.modifier_apply(modifier='Solidify')

bpy.ops.export_scene.obj(filepath=obj_path.replace('.obj', '_after.obj'), use_materials=False, use_triangles=True)
bpy.ops.export_mesh.ply(filepath=ply_path)
bpy.ops.wm.quit_blender()
""",
        "--", obj_filename, ply_filename])

def train(rank, world_size, batch_size):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(rank)
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    output_dir = os.path.abspath(os.path.expanduser("~/shap-e/shap_e/tests/generated_3d_models"))
    os.makedirs(output_dir, exist_ok=True)
    dist.barrier()
    
    xm = robust_load_model("transmitter", device)
    xm = torch.nn.parallel.DistributedDataParallel(xm, device_ids=[rank], output_device=rank)
    
    local_batch_size = max(1, (batch_size + world_size - 1 - rank) // world_size)
    
    if local_batch_size > 0:
        model = robust_load_model("text300M", device)
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
    
    for i, latent in enumerate(latents):
        obj_before = get_next_filename(output_dir, ".obj")
        obj_after = obj_before.replace(".obj", "_after.obj")
        ply_after = obj_before.replace(".obj", "_after.ply")
        gif_before = obj_before.replace(".obj", ".gif")
        gif_after = obj_after.replace(".obj", ".gif")

        mesh = decode_latent_mesh(xm.module, latent)
        tri_mesh = mesh.tri_mesh()
        with open(obj_before, "w") as f:
            tri_mesh.write_obj(f)

        generate_gif(xm, latent, gif_before, device)
        clean_and_thicken_mesh_in_blender(obj_before, ply_after)
        generate_gif(xm, latent, gif_after, device)
    
    dist.barrier()
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    batch_size = int(os.getenv("BATCH_SIZE", 2))
    mp.spawn(train, args=(world_size, batch_size), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()