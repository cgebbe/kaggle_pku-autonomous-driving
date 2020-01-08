

# About

- This is my personal code for the kaggle competition https://www.kaggle.com/c/pku-autonomous-driving
- It can be run using the main.py file
- It uses python3 and the default kaggle python libraries (default google colab libraries seems to work as well)

# Log

| Changes                                                      | Result                                           | Ideas for next steps                                         |
| ------------------------------------------------------------ | ------------------------------------------------ | ------------------------------------------------------------ |
| Implemented notebook https://www.kaggle.com/phoenix9032/center-resnet-starter | public LB 0.020 - seems very low, place ~660/800 | Test more popular notebook by Rusian                         |
| Implemented notebook https://www.kaggle.com/hocop1/centernet-baseline | public LB 0.031 - place 550/800 -> better start  | - implement image augmentation during training <br />- use masks<br />- sin/cos(roll) ?<br />- higher model resolution<br />- calculate KPI yourself -> different loss ? () |
|                                                              |                                                  |                                                              |

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
  
# Idea collection
- copy optimize x,y,z - uv  
- image augmentation
- use masks
- calc LB yourself -> different loss ?
- instead x,y,z estimate (u,v),depth! Maybe much simpler? 
- increase image size
