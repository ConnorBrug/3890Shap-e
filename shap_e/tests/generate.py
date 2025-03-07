import os
import time
import torch
import glob
import imageio
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import (
    create_pan_cameras,
    decode_latent_images,
    decode_latent_mesh,
)

# Suppress ALSA errors (audio system)
os.environ["AUDIODEV"] = "null"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PULSE_SERVER"] = ""
os.environ["ALSA_CARD"] = "0"
os.environ["ALSA_PCM_CARD"] = "0"

def robust_load_model(model_name, device, max_attempts=5, delay=10):
    """Tries to load a model multiple times to prevent crashes due to download issues."""
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
    """Generates a GIF preview of the latent if it has valid data."""
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

def get_next_folder(output_dir):
    """Generate the next sequential folder like object_1, object_2, etc."""
    existing_folders = glob.glob(os.path.join(output_dir, "object_*"))
    numbers = [int(f.split("_")[-1]) for f in existing_folders if f.split("_")[-1].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1
    folder_name = os.path.join(output_dir, f"object_{next_number}")
    os.makedirs(folder_name, exist_ok=True)
    return folder_name, next_number

def clean_and_thicken_mesh_in_blender(obj_filename, obj_after_filename):
    """Runs Blender in headless mode to fix thin and disconnected parts."""
    blender_path = os.path.expanduser("~/software/blender/blender")  # Adjust Blender path if needed

    blender_script = f"""
import bpy
import sys
import os

obj_path = sys.argv[-2]
obj_after_path = sys.argv[-1]

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Check if the .obj file exists before importing
if not os.path.exists(obj_path):
    print(f"Error: OBJ file not found: {{obj_path}}")
    bpy.ops.wm.quit_blender()

# Import OBJ with error handling
try:
    bpy.ops.import_scene.obj(filepath=obj_path, use_split_groups=False)
    mesh = bpy.context.selected_objects[0]
except Exception as e:
    print(f"Blender Import Error: {{e}}")
    bpy.ops.wm.quit_blender()

# Set the object as active
bpy.context.view_layer.objects.active = mesh

# Ensure we are in Object Mode before applying modifiers
bpy.ops.object.mode_set(mode='OBJECT')

# Step 1: Fix any mesh issues (Recalculate Normals)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.01)  # Merge close vertices
bpy.ops.mesh.fill_holes(sides=6)  # Fill large gaps
bpy.ops.mesh.normals_make_consistent(inside=False)  # Fix flipped normals
bpy.ops.object.mode_set(mode='OBJECT')  # Back to Object Mode

# Step 2: Thicken Super Thin Parts
solidify_mod = mesh.modifiers.new(name="Solidify", type='SOLIDIFY')
solidify_mod.thickness = 0.05  # Increased thickness for a visible effect
solidify_mod.offset = 1.0  
solidify_mod.use_rim = True  # Ensures edges remain connected

# Apply the Solidify Modifier
bpy.ops.object.modifier_apply(modifier="Solidify")

# Apply all transformations
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Export final OBJ
bpy.ops.export_scene.obj(filepath=obj_after_path, use_materials=False, use_triangles=True)
bpy.ops.wm.quit_blender()
"""

    subprocess.run([blender_path, "--background", "--python-expr", blender_script, "--", obj_filename, obj_after_filename])

def train(rank, world_size, batch_size):
    """Distributed processing for latent generation and 3D model decoding."""
    print(f"[Rank {rank}] Process started.")
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

    # Ensure all batch items are accounted for
    local_batch_size = batch_size // world_size + (1 if rank < batch_size % world_size else 0)

    print(f"[Rank {rank}] Assigned batch size: {local_batch_size}")

    latents = None
    if local_batch_size > 0:
        print(f"[Rank {rank}] Generating {local_batch_size} latents...")
        try:
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
        except Exception as e:
            print(f"[Rank {rank}] Error generating latents: {e}")
            latents = None  # Ensure `latents` is None if an error occurs

    if latents is None or (isinstance(latents, torch.Tensor) and latents.numel() == 0):
        print(f"[Rank {rank}] No latents assigned! Skipping...")
        dist.barrier()
        dist.destroy_process_group()
        return

    for i, latent in enumerate(latents):
        # Create a folder for this object
        object_folder, object_number = get_next_folder(output_dir)
        obj_before = os.path.join(object_folder, f"object_{object_number}_before.obj")
        obj_after = os.path.join(object_folder, f"object_{object_number}_after.obj")
        gif_before = os.path.join(object_folder, f"object_{object_number}_before.gif")
        gif_after = os.path.join(object_folder, f"object_{object_number}_after.gif")

        print(f"[Rank {rank}] Processing object {i} in folder: {object_folder}")

        # Decode latent to mesh
        mesh = decode_latent_mesh(xm.module, latent)
        if mesh is None or not hasattr(mesh, 'tri_mesh'):
            print(f"[Rank {rank}] Skipping invalid mesh for object {i}.")
            continue

        tri_mesh = mesh.tri_mesh()
        if not hasattr(tri_mesh, 'verts') or not hasattr(tri_mesh, 'faces'):
            print(f"[Rank {rank}] Skipping invalid tri_mesh for object {i}.")
            continue

        # Export .obj before Blender
        with open(obj_before, "w") as f:
            tri_mesh.write_obj(f)

        # Generate GIF for pre-Blender .obj
        generate_gif(xm, latent, gif_before, device)

        # Process in Blender
        clean_and_thicken_mesh_in_blender(obj_before, obj_after)

        # Generate GIF for post-Blender .obj
        generate_gif(xm, latent, gif_after, device)

    dist.barrier()
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    batch_size = int(os.getenv("BATCH_SIZE", 2))
    mp.spawn(train, args=(world_size, batch_size), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()