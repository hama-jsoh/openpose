import json
import numpy as np
import cv2


KEYPOINTS_NAME = [
    "Nose",
    "Neck",
    "R-Sho",
    "R-Elb",
    "R-Wr",
    "L-Sho",
    "L-Elb",
    "L-Wr",
    "R-Hip",
    "R-Knee",
    "R-Ank",
    "L-Hip",
    "L-Knee",
    "L-Ank",
    "R-Eye",
    "L-Eye",
    "R-Ear",
    "L-Ear",
]


class OpenPose:
    def __init__(
        self,
        model,
        verbose: bool = True,
    ) -> None:
        if model is not None:
            if model == "coco":
                self.protoFile = "./pretrained_models/pose_deploy_linevec.prototxt"
                self.weightFile = "./pretrained_models/pose_iter_440000.caffemodel"
                self.nPoints = 18
        else:
            raise Exception("model required!, [recommends: 'coco']")
        self.verbose = verbose

    def GetKeypoints(self, probMap, threshold=0.1) -> list:
        mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
        mapMask = np.uint8(mapSmooth > threshold)

        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, _, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints = list(maxLoc + (round(float(probMap[maxLoc[1], maxLoc[0]]), 6),))
        return keypoints

    def Inference(self, filename: str = "messi.jpg") -> dict:
        net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightFile)
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        if self.verbose:
            print("Using CPU device")

        image = cv2.imread(f"./data/test/{filename}")
        frameWidth = image.shape[1]
        frameHeight = image.shape[0]
        inHeight = 368
        inWidth = int((inHeight / frameHeight) * frameWidth)
        inpBlob = cv2.dnn.blobFromImage(
            image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False
        )

        net.setInput(inpBlob)
        output = net.forward()

        detected_keypoints = []
        threshold = 0.1
        person_id = [-1]

        for part in range(self.nPoints):
            probMap = output[0, part, :, :]
            probMap = cv2.resize(probMap, (image.shape[1], image.shape[0]))
            keypoints = self.GetKeypoints(probMap, threshold)
            if self.verbose:
                print("Keypoints - {} : {}".format(KEYPOINTS_NAME[part], keypoints))
            detected_keypoints.append(keypoints)

        kpts = []
        for kpt in detected_keypoints:
            x, y, p = kpt
            kpts.append(x)
            kpts.append(y)
            kpts.append(p)

        people = [
            dict(
                person_id=person_id,
                pose_keypoints_2d=kpts,
                face_keypoints_2d=[],
                hand_left_keypoints_2d=[],
                hand_right_keypoints_2d=[],
                pose_keypoints_3d=[],
                face_keypoints_3d=[],
                hand_left_keypoints_3d=[],
                hand_right_keypoints_3d=[],
            )
        ]

        json_format = dict(version=1.3, people=people)
        return json_format

    @staticmethod
    def FileOutput(dict_obj, path: str, indent: bool = False) -> None:
        with open(path, "w") as j:
            if indent:
                json.dump(dict_obj, j, ensure_ascii=False, indent=4)
            else:
                json.dump(dict_obj, j, ensure_ascii=False)


if __name__ == "__main__":

    # openpose configuration
    pose = OpenPose(
        model="coco",
        verbose=False,
    )

    # run openpose
    kpts = pose.Inference()

    # fileio
    pose.FileOutput(
        dict_obj=kpts,
        path="./output.json",
        indent=False,
    )
