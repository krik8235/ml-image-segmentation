# Build an Image Segmentation Model for Medical Diagnosis using CNN, PyTorch and OpenCV

In healthcare and medical science, the fusion of artificial intelligence and deep learning is revolutionizing diagnostics. My project focuses on the precise segmentation of polyps from colonoscopy images—a vital tool for medical practitioners.

In this project, we aim to develop an image segmentation model using CNN, and deploy it with PyTorch.



## Data

The use of the CVC-Clinic database, containing frames from colonoscopy videos. The dataset includes polyp frames and corresponding ground truth images in both PNG and TIFF formats.


## Tech stack

- Python 3.12.7
- **Deep Learning**: PyTorch
- **Computer Vision**: OpenCV (python-opencv)
- **Libraries**: Scikit-learn, Pandas, NumPy, Albumentations, YAML (data serialization)


## Start in the local env

* Start with installing jupyter environment and dependencies.
```
$ pip install ipykernel
$ pip install -r requirements.txt
```

* Folder structure
  - .venv
  - dataset
  - ml_deploy
  - output



## Takeaways
After 100 epochs of training, 
![Actual image]('https://github.com/krik8235/ml-image-segmentation/blob/main/output/test-images/act.png?raw=true')
![Segmentation image]('https://github.com/krik8235/ml-image-segmentation/blob/main/output/test-images/seg.png?raw=true')


- Polyp Segmentation Insights
- IOU Metric Understanding
- Data Augmentation Techniques
- Practical PyTorch Data Augmentation
- Medical Computer Vision Applications
- Building CNN Models
- Training CNN Models (300 epochs)

