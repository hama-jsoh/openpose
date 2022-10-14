# Simple OpenPose

## Getting Started

### Pre-requisite

1. Download pretrained models : COCO > [Google-Drive](https://drive.google.com/drive/folders/1Oz_fDTMDSttZMu-Va6kGvE81kaggm3sC?usp=sharing)

```bash
pretrained_models/
├── pose_deploy_linevec.prototxt
└── pose_iter_440000.caffemodel
```

2. Put data
```
data/
└── human
    ├── human_00.jpg
    ├── human_01.jpg
    └── ...
```

3. Run
```bash
python3 openpose.py
```
  
#
### Simple Example
```python
import openpose


if __name__ == "__main__":

    # openpose configuration
        pose = OpenPose(
		model="coco",
		verbose=True,
	)

    # run openpose
    kpts = pose.Inference(dataroot="./data/human")

    # fileio
    pose.FileOutput(
	dict_obj=kpts,
        indent=False,
    )
```
