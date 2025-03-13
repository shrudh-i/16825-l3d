import subprocess 

def main():
    '''
    1. Exploring Loss Functions
    '''
    #1.1. Fitting a voxel grid (5 points)
    subprocess.run(["python", "fit_data.py", "--type", "vox"])
    #1.2. Fitting a point cloud (1000 points)
    subprocess.run(["python", "fit_data.py", "--type", "point"])
    #1.3. Fitting a mesh (5 points)
    subprocess.run(["python", "fit_data.py", "--type", "mesh"])

    '''
    2. Reconstructing 3D from a single view:
        # Voxels: --max_iter 30_000 --save_freq 1000
        # Point:  --max_iter 10_000 --save_freq 1000
        # Mesh:   --max_iter 10_000 --save_freq 1000
    '''
    
    # To train the models for each of the three types of data, please uncomment lines 24, 31, 38

    # Q2.1
    # subprocess.run(["python", "train_model.py", "--type", "vox", "--max_iter", "20_000", "--save_freq", "500"])
    subprocess.run(["python", "eval_model.py", "--type", "vox", "--load_checkpoint"])
    
    # For selecting a specific instances of the dataset to work with, please uncomment the following line:
    # subprocess.run(["python", "eval_model.py", "--type", "vox", "--load_checkpoint", "--vis_freq", "50"])

    # Q2.2
    # subprocess.run(["python", "train_model.py", "--type", "point", "--max_iter", "10_000", "--save_freq", "500"])
    subprocess.run(["python", "eval_model.py", "--type", "point", "--load_checkpoint"])

    # For selecting a specific instances of the dataset to work with, please uncomment the following line:
    # subprocess.run(["python", "eval_model.py", "--type", "point", "--load_checkpoint", "--vis_freq", "50"])
    
    # Q2.3
    # subprocess.run(["python", "train_model.py", "--type", "mesh",  "--max_iter", "10_000", "--save_freq", "500"])
    subprocess.run(["python", "eval_model.py", "--type", "mesh", "--load_checkpoint"])

    # For selecting a specific instances of the dataset to work with, please uncomment the following line:
    # subprocess.run(["python", "eval_model.py", "--type", "mesh", "--load_checkpoint", "--vis_freq", "50"])

    # Q2.5
    # Before running these lines of code, please rename the .pth files from training before to save them according to the hyperparams we are changing:
    '''
    # Mesh - w_smooth=0.01
    subprocess.run(["python", "train_model.py", "--type", "vox", "--max_iter", "20_000", "--save_freq", "500", "--w_smooth", "0.01"])
    subprocess.run(["python", "eval_model.py", "--type", "vox", "--load_checkpoint", "--w_smooth", "0.01"])

    # Mesh - w_smooth=2.0
    subprocess.run(["python", "train_model.py", "--type", "vox", "--max_iter", "20_000", "--save_freq", "500", "--w_smooth", "2.0"])
    subprocess.run(["python", "eval_model.py", "--type", "vox", "--load_checkpoint", "--w_smooth", "2.0"])

    # Mesh - w_smooth=4.0
    subprocess.run(["python", "train_model.py", "--type", "vox", "--max_iter", "20_000", "--save_freq", "500", "--w_smooth", "4.0"])
    subprocess.run(["python", "eval_model.py", "--type", "vox", "--load_checkpoint", "--w_smooth", "4.0"])
    '''
    
    # Q2.6
    subprocess.run(["python", "eval_model.py", "--type", "point", "--load_checkpoint", "--vis_2_6", "True"])
    # To visualize different instances of the dataset, uncomment the following line:
    # subprocess.run(["python", "eval_model.py", "--type", "point", "--load_checkpoint", "--vis_2_6", "True", "vis_freq", "50"])

    '''
    2. Reconstructing 3D from a single view:
        # Voxels: --max_iter 30_000 --save_freq 1000
        # Point:  --max_iter 10_000 --save_freq 1000
        # Mesh:   --max_iter 10_000 --save_freq 1000
    '''
    # Q3.3
    # Before running this question, set "using the full dataset" in dataset_location.py
    # ["python", "train_model.py", "--type", "mesh"]
    # ["python", "eval_model.py", "--type", "mesh", "--load_checkpoint"]


if __name__ == "__main__":
    main()