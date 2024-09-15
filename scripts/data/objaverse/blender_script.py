import argparse
import math
import os
import random
import shutil
import sys
import time
import urllib.request
from mathutils import Vector
import numpy as np
import bpy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./rendering_random_32views")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=32)
parser.add_argument("--resolution", type=int, default=1024)
parser.add_argument("--gpu_id", type=int, default=0)

if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = sys.argv[1:]

args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

# Set the device_type
cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
cycles_preferences.compute_device_type = "CUDA"
cuda_devices = cycles_preferences.get_devices_for_type("CUDA")
for device in cuda_devices:
    device.use = False


def compose_RT(R, T):
    return np.hstack((R, T.reshape(-1, 1)))


def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def set_camera_location(camera, option: str):
    assert option in ['fixed', 'random', 'front']

    if option == 'fixed':
        x, y, z = 0, -2.25, 0
    elif option == 'random':
        x, y, z = sample_spherical(radius_min=1.9, radius_max=2.6, maxz=1.60, minz=-0.75)
    elif option == 'front':
        x, y, z = 0, -np.random.uniform(1.9, 2.6, 1)[0], 0

    camera.location = x, y, z
    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def add_lighting(option: str) -> None:
    assert option in ['fixed', 'random']

    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()

    bpy.ops.object.light_add(type="AREA")
    light = bpy.data.lights["Area"]

    if option == 'fixed':
        light.energy = 30000
        bpy.data.objects["Area"].location[0] = 0
        bpy.data.objects["Area"].location[1] = 1
        bpy.data.objects["Area"].location[2] = 0.5

    elif option == 'random':
        light.energy = random.uniform(80000, 120000)
        bpy.data.objects["Area"].location[0] = random.uniform(-2., 2.)
        bpy.data.objects["Area"].location[1] = random.uniform(-2., 2.)
        bpy.data.objects["Area"].location[2] = random.uniform(1.0, 3.0)

    bpy.data.objects["Area"].scale[0] = 200
    bpy.data.objects["Area"].scale[1] = 200
    bpy.data.objects["Area"].scale[2] = 200


def reset_scene() -> None:
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene(box_scale: float):
    bbox_min, bbox_max = scene_bbox()
    scale = box_scale / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 24
    cam.data.sensor_width = 32
    cam.data.sensor_height = 32
    cam.data.clip_start = 0.01
    cam.data.clip_end = 6.0
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def save_images(object_file: str) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene(box_scale=2)
    add_lighting(option='random')
    camera, cam_constraint = setup_camera()

    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    img_dir = os.path.join(args.output_dir, object_uid)
    os.makedirs(img_dir, exist_ok=True)

    # Prepare to save camera parameters
    cam_params = {
        "intrinsics": get_calibration_matrix_K_from_blender(camera.data, return_principles=True),
        "cam_poses": [],
        "cam_poses_w2c": []
    }

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for n in tree.nodes:
        tree.nodes.remove(n)

    bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_z = True
    rl = tree.nodes.new(type='CompositorNodeRLayers')

    # Save images
    image_save = tree.nodes.new(type='CompositorNodeOutputFile')
    links.new(rl.outputs['Image'], image_save.inputs['Image'])

    # Save depth maps as png and exr
    depth_map = tree.nodes.new(type="CompositorNodeMapRange")
    depth_map.inputs['From Min'].default_value = 0.01
    depth_map.inputs['From Max'].default_value = 6.0
    depth_map.inputs['To Min'].default_value = 0.0
    depth_map.inputs['To Max'].default_value = 1.0
    links.new(rl.outputs['Depth'], depth_map.inputs['Value'])
    depth_save = tree.nodes.new(type="CompositorNodeOutputFile")
    links.new(depth_map.outputs[0], depth_save.inputs['Image'])

    depth_save_exr = tree.nodes.new(type="CompositorNodeOutputFile")
    links.new(rl.outputs['Depth'], depth_save_exr.inputs['Image'])

    # Save normals
    normal_save = tree.nodes.new(type="CompositorNodeOutputFile")
    links.new(rl.outputs['Normal'], normal_save.inputs['Image'])

    for i in range(args.num_images):
        image_save.base_path = img_dir
        image_save.file_slots[0].use_node_format = True
        image_save.file_slots[0].path = f"{i:03d}"
        image_save.format.file_format = 'PNG'
        image_save.format.color_mode = 'RGBA'

        depth_save.base_path = img_dir
        depth_save.file_slots[0].use_node_format = True
        depth_save.file_slots[0].path = f"{i:03d}_depth"
        depth_save.format.file_format = 'PNG'
        depth_save.format.color_mode = 'BW'
        depth_save.format.color_depth = '8'

        depth_save_exr.base_path = img_dir
        depth_save_exr.file_slots[0].use_node_format = True
        depth_save_exr.file_slots[0].path = f"{i:03d}_depth"
        depth_save_exr.format.file_format = 'OPEN_EXR'
        depth_save_exr.format.color_depth = '32'

        normal_save.base_path = img_dir
        normal_save.file_slots[0].use_node_format = True
        normal_save.file_slots[0].path = f"{i:03d}_normal"
        normal_save.format.file_format = 'PNG'
        normal_save.format.color_mode = 'RGBA'

        # Set the camera position
        camera_option = 'random' if i > 0 else 'front'
        camera = set_camera_location(camera, option=camera_option)
        bpy.ops.render.render(write_still=True)

        # Save camera RT matrix (C2W)
        location, rotation = camera.matrix_world.decompose()[0:2]
        RT = compose_RT(rotation.to_matrix(), np.array(location))
        cam_params["cam_poses"].append(RT)

        # Save camera RT matrix (W2C)
        RT_2 = get_3x4_RT_matrix_from_blender(camera)
        cam_params["cam_poses_w2c"].append(RT_2)

    for file_name in os.listdir(img_dir):
        if os.path.isfile(os.path.join(img_dir, file_name)):
            name, extension = os.path.splitext(file_name)
            new_name = name[:-4]
            new_filename = new_name + extension

            print(name, extension, new_name, new_filename)
            old_filepath = os.path.join(img_dir, file_name)
            new_filepath = os.path.join(img_dir, new_filename)
            shutil.move(old_filepath, new_filepath)

    # Save camera intrinsics and poses
    np.savez(os.path.join(img_dir, 'cameras.npz'), **cam_params)


def download_object(object_url: str) -> str:
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    local_path = os.path.abspath(local_path)
    return local_path

def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix())
    t = np.asarray(location)

    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)
    return RT

def get_calibration_matrix_K_from_blender(camera, return_principles=False):
    render = bpy.context.scene.render
    width = render.resolution_x * render.pixel_aspect_x
    height = render.resolution_y * render.pixel_aspect_y
    focal_length = camera.lens
    sensor_width = camera.sensor_width
    sensor_height = camera.sensor_height
    focal_length_x = width * (focal_length / sensor_width)
    focal_length_y = height * (focal_length / sensor_height)
    optical_center_x = width / 2
    optical_center_y = height / 2
    K = np.array([[focal_length_x, 0, optical_center_x],
                  [0, focal_length_y, optical_center_y],
                  [0, 0, 1]])
    if return_principles:
        return np.array([
            [focal_length_x, focal_length_y],
            [optical_center_x, optical_center_y],
            [width, height],
        ])
    else:
        return K

def main(args):
    try:
        start_i = time.time()
        local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)


if __name__ == "__main__":
    main(args)
