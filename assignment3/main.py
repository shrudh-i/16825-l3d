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
    # TODO: add the command to run this
    '''

    '''
    5. Sphere Tracing
    '''
    subprocess.run(["python", "-m", "surface_rendering_main", "--config-name", "torus_surface"])


if __name__ == "__main__":
    main()