import subprocess 

def main():
    '''
    1. Differentiable Volume Rendering
    '''
    # subprocess.run(["python", "volume_rendering_main.py", "--config-name", "box"])

    '''
    2. Optimizing a basic implicit volume
    '''
    # subprocess.run(["python", "volume_rendering_main.py", "--config-name", "train_box"])

    '''
    3. Optimizing a Neural Radiance Field (NeRF)
    '''
    # subprocess.run(["python", "volume_rendering_main.py", "--config-name", "nerf_lego"])

    '''
    4. NeRF Extras
    '''
    # subprocess.run(["python", "volume_rendering_main.py", "--config-name", "nerf_materials"])
    # subprocess.run(["python", "volume_rendering_main.py", "--config-name", "nerf_materials_highres"])

    '''
    5. Sphere Tracing
    '''
    # subprocess.run(["python", "-m", "surface_rendering_main", "--config-name", "torus_surface"])

    '''
    6. Optimizing a Neural SDF
    '''
    # subprocess.run(["python", "-m", "surface_rendering_main", "--config-name", "points_surface"])

    '''
    7. VolSDF
    '''
    # subprocess.run(["python", "-m", "surface_rendering_main", "--config-name", "volsdf_surface"])

    '''
    8.1. Render a Large Scene with Sphere Tracing
            * Make sure to uncomment the necessary lines in `surface_rendering_main.py` & 'implicit.py' to enable the large scene rendering.
    '''
    # subprocess.run(["python", "-m", "surface_rendering_main", "--config-name", "custom_surface"])



if __name__ == "__main__":
    main()