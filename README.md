

# About

- This is my personal code for the kaggle competition https://www.kaggle.com/c/pku-autonomous-driving
- It can be run using the main.py file
- It uses python3 and the default kaggle python libraries (default google colab libraries seems to work as well)

# Log

| Changes                                                      | Result                             | Ideas for next steps                                         |
| ------------------------------------------------------------ | ---------------------------------- | ------------------------------------------------------------ |
| Run notebook https://www.kaggle.com/phoenix9032/center-resnet-starter | public LB 0.020, place ~660/800    | Seems low place ->Test more popular notebook by Rusian       |
| Run notebook https://www.kaggle.com/hocop1/centernet-baseline | public LB 0.031, place 550/800     | Better start, seems okay as a baseline. -> Implement prediction pipeline yourself |
| Own implementation of previous notebook (in particular data loading and inference pipeline). Only use model weights by hocop1 / ruslan | public LB 0.006                    | something is still wrong, despite predicted images seem okay :/ -> compare code to original notebook |
| Fixed stuff in reimplementation and integrated optimize_xy function. Now, prediction is exactly equal using original weights | public LB 0027                     | Should also score 0.031 ?  -> Try with my weights and see what happens |
| Exact same using my weights                                  | public LB 0.034, place 502/810     | surprisingly large difference. -> Next: Evaluate improvement idea collection and start gradually |
| Increased model input size from 1024x320 to 1536x512 (without modifying layers. Maybe another convolution would be necessary, because effective window size decreases?). Added another upsample convolution to increase output size to 384x128 (before 128x40) | x - aborted training after epoch 4 | mask loss dominates. -> Change weights s.t. mask and regr loss are similar |
| Changed weights, so that mask and regr loss are same order of magnitude, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/115673 | public LB 0.062, place 109/820     | :)  -> Next: choose next improvement idea                    |
| Major change: Switched from binary loss to focal loss for mask. Minor change: Excluded five erroneous images from training set | public LB 0                        | :( Mask seems great, but regression values are totally wrong |
| In training: Changed regression loss by extracting binary mask from heatmap mask. Also changed learning rate scheduler slighlier. In prediction post-processing: Disable optimization if optimized values don't make any sense | public LB                          |                                                              |
| Added image augmentation in training (most importantly hor flip). MAp is now calculated during training to assess model quality |                                    |                                                              |

# Improvement idea collection
- larger image size 1536*512
  - together with focal loss achieves LB 0.078, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/123193
- change to focal loss (?)
  - https://www.kaggle.com/c/pku-autonomous-driving/discussion/121608
  - https://www.kaggle.com/c/pku-autonomous-driving/discussion/115673
  - heatmap, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/123090
  - use AdamW optimizer, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/121608
- exclude broken images in training, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/117621
  - ID_1a5a10365.jpg
  - ID_4d238ae90.jpg
  - ID_408f58e9f.jpg
  - ID_bb1d991f6.jpg
  - ID_c44983aeb.jpg
- model backbone. Options:
  - efficientnet b0 (LB 0.078 with above changes)
  - Hourglass104
  - DLA34
- image augmentation: horizontal flip, random brightness, gaussian noise and contrast in training. -> only slight improvement
  - Pay attention to horizontal flipping, not so easy, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/125591
- use masks, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/122942
- calc LB yourself and use it to evaluate model after each epoch
  - https://www.kaggle.com/c/pku-autonomous-driving/discussion/117578
  - https://www.kaggle.com/c/pku-autonomous-driving/discussion/124870
- instead x,y,z estimate (u,v),depth! Maybe much simpler task? But requires significant restructuring
- include test set in training via pseudo labels, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/124912
- use more than one model and ensemble them
- In general a detailed error analysis. For example
  - What is the kpi on near / distant cars?
  - Which thresholds are reached, which not? Is the limiting factor the rotational or the translational error?



# Lessons learned

- Simply download the predictions.csv output file from a notebook to evaluate its score instead of rerunning the whole notebook and waiting 12h
- Instead of starting code from scratch rather refactor the existing notebook code. While starting from scratch is a greater learning experience and produces more structured code (in my view), it is quite difficult and cumbersome to get all details right in the reimplementation and thus have a defined starting point.
- Buy a graphic card, which can at least perform inference with the desired model. Solution now was to use a google colab GPU via ssh and pycharm remote (possible through ngrok "hack"). However, this is unstable and time tedious. I believe I could have run 2x as many trainings using a decent desktop PC.



# Idea for more efficient GPU use

- Use local GPU and a tiny model (e.g. 1/8 of channels) to test code locally
- Use 2 GPUs for actual training (so that I can train and test new stuff in parallel !):
  - google colab
    - data -> download from kaggle into gdrive once
    - code -> transfer via ssh to gdrive (or via web to gdrive). Start trainings from within notebook, so that local GPU free. 
  - kaggle
    - data -> already in kaggle
    - code -> transfer via github.
      - Or can I also mount gdrive? Would be even simpler! -> Seems complicated, see below. Probably not worth, given that the GPU quota limits training to 30h / week anyhow, which means ~ 2 trainings / week
        - https://developers.google.com/drive/api/v3/quickstart/python
        - https://towardsdatascience.com/how-to-manage-files-in-google-drive-with-python-d26471d91ecd
- github vs ssh
  - github
    - (+) quick to setup
    - (-) cannot debug or set breakpoints
  - ssh + pycharm remote
    - (-) need to run ssh, ngrok and click event. Potentially more unstable?
    - (+) code is copied instantaneously
- -> add additional params (to overwrite) via argparse
  - location of data
  - flag_simplify_model (only for local running!)

# Why does google colab session crash?

- see log in /var/log/colab-jupyter.log

