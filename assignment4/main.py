import subprocess 

def main():
    '''
    1.1.5 Perform Splatting
    '''
    # subprocess.run(["python", "render.py", "--gaussians_per_splat", "1024"])

    '''
    1.2.2 Perform Forward Pass and Compute Loss
    '''
    # subprocess.run(["python", "train.py", "--gaussians_per_splat", "1024", "--num_itrs", "1000", "-checkpoint_freq", "200"])

    '''
    1.3.1 Rendering Using Spherical Harmonics
    '''
    #NOTE: For SDS guidance, ensure to fill in the prompt with what you want to generate:
    # subprocess.run(["python", "Q21_image_optimization.py", "--sds_guidance", "1", "--prompt", "a koi pond"])

    #NOTE: For SDS guidance, ensure to fill in the prompt with what you want to generate:
    # subprocess.run(["python", "Q21_image_optimization.py", "--sds_guidance", "0", "--prompt", "a bonsai tree"])

    '''
    2.2 Texture Map Optimization for Mesh 
    '''
    #NOTE: Replace the prompt with what you would like the texture to be before running:
    # subprocess.run(["python", "Q22_nerf_optimization.py", "--prompt", "a zebra"])

    '''
    2.3 NeRF Optimization 
    '''
    # subprocess.run(["python", "Q23_nerf_optimization.py", "--prompt", "a standing corgi dog"])

    '''
    6. Optimizing a Neural SDF
    '''
    # subprocess.run(["python", "Q23_nerf_optimization.py", "--prompt", "a standing corgi dog", "--view_dep_texture", "1"])



if __name__ == "__main__":
    main()