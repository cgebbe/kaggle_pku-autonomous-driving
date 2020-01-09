

# About

- This is my personal code for the kaggle competition https://www.kaggle.com/c/pku-autonomous-driving
- It can be run using the main.py file
- It uses python3 and the default kaggle python libraries (default google colab libraries seems to work as well)

# Log

| Changes                                                      | Result                          | Ideas for next steps                                         |
| ------------------------------------------------------------ | ------------------------------- | ------------------------------------------------------------ |
| Run notebook https://www.kaggle.com/phoenix9032/center-resnet-starter | public LB 0.020, place ~660/800 | Seems low place ->Test more popular notebook by Rusian       |
| Run notebook https://www.kaggle.com/hocop1/centernet-baseline | public LB 0.031, place 550/800  | Better start, seems okay as a baseline. -> Implement prediction pipeline yourself |
| Own implementation of previous notebook (in particular data loading and inference pipeline). Only use model weights by hocop1 / ruslan | public LB 0.006                 | something is still wrong, despite predicted images seem okay :/ -> compare code to original notebook |
| Fixed stuff in reimplementation and integrated optimize_xy function. Now, prediction is exactly equal using original weights | public LB 0027                  | Should also score 0.031 ?  -> Try with my weights and see what happens |

# Todo
- ensure that prediction output equals ruslan output
  - ensure item->mat works 
    - resizing -> yes
    - order 8 channels of mat -> yes
  - logits / confidence 
    - is threshold correct? -> yes
    - is 1/(1+exp(-c)) correct? -> yes
  - why is cos/sin on pitch/yaw necessary?
    -> on yaw, because can be [-pi, pi] and similar values
  - image reading same? -> yes
- Why are two cars instead of one detected?!
  - image reading RGB /BGR ? -> correct
  - conversion mat -> item --> check logits directly -> correct
  - optimize function -> integrated. Now results exactly the same
  - Why is image so noisy? Seems really weird... 
  - I believe that results are slightly different due to 
# Improvement idea collection
- larger image size 1536*512
  - together with focal loss achieves LB 0.078, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/123193
- change to focal loss (?)
  - https://www.kaggle.com/c/pku-autonomous-driving/discussion/121608
  - https://www.kaggle.com/c/pku-autonomous-driving/discussion/115673
- exclude broken images in training, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/117621
  - ID_1a5a10365.jpg
  - ID_4d238ae90.jpg
  - ID_408f58e9f.jpg
  - ID_bb1d991f6.jpg
  - ID_c44983aeb.jpg
- model backbone
  - efficientnet b0 (LB 0.078 with above changes)
  - Hourglass104
  - DLA34
- model 
- image augmentation: horizontal flip, random brightness, gaussian noise and contrast in training. -> only slight improvement
- use masks, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/122942
- calc LB yourself -> different loss ? see https://www.kaggle.com/c/pku-autonomous-driving/discussion/117578
- instead x,y,z estimate (u,v),depth! Maybe much simpler task? But requires significant restructuring
- increase image size
- include test set in training via pseudo labels, see https://www.kaggle.com/c/pku-autonomous-driving/discussion/124912



# Lessons learned

- Simply download the predictions.csv output file from a notebook to evaluate its score instead of rerunning the whole notebook and waiting 12h
- Instead of starting code from scratch rather refactor the existing notebook code. While starting from scratch is a greater learning experience and produces more structured code (in my view), it is quite difficult and cumbersome to get all details right in the reimplementation and thus have a defined starting point.
- Buy a graphic card, which can at least perform inference with the desired model. Solution now was to use a google colab GPU via ssh and pycharm remote (possible through ngrok "hack"). However, this is unstable and time tedious...