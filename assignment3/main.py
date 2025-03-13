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
    subprocess.run(["python", "volume_rendering_main.py", "--config-name", "nerf_lego"])



if __name__ == "__main__":
    main()