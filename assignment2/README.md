# Instructions for Running Homework-1:
All deliverables required for Homework-1 can be obtained by running:

```
python main.py
```
All output files will be generated into the _submission_ directory, please ensure not to modify or remove it.

Note: if hyperparameter outputs do not show up in the _submission_ directory, please create a folder according to the file naming conventions in `eval_model.py`. 

For example: hyperparam testing with _w_smooth=0.01_ would result in _'mesh_eval_0.01'_

# Important Notes on Implementation:
* All training done for this assignment were performed using personal laptop GPU and not using AWS
* I have added a `utils.py` file for my visualization functions. I call those accordingly for each section
* The `main.py` file contains all the terminal calls required to run all sections for Assignment-2
* For section 2.6, please use the argument I have added `--vis_2_6`. Setting it to true, generates an output pertaining to that section in the _eval_point/submission_ folder
* All weights _(.pth files)_ are present within the _weights_ folder, ensure to move them to the main directory `~/assignment2/` before running the pertaining section
* Please make note of the comments in the `main.py` file, the lines on training the model have been commented out of interest for submission. If required, please uncomment to train certain lines 