# Simple OpenPose

## Getting Started

### Step 1: Set Up the Folder Structure (models)

- Pretrained models : [COCO (Google-Drive)](https://drive.google.com/drive/folders/1Oz_fDTMDSttZMu-Va6kGvE81kaggm3sC?usp=sharing)

```bash
pretrained_models/
├── pose_deploy_linevec.prototxt
└── pose_iter_440000.caffemodel
```
or  

- Just, Run (Automatic setting)
```bash
bash setup.sh
```


### Step 2: Put data
```
data/
└── human
    ├── human_00.jpg
    ├── human_01.jpg
    └── ...
```

### Step 3: Run
```bash
python3 openpose.py
```
  
### Step 4: Check output
```bash
ls data/samples
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
