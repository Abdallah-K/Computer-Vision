import os
import cv2
import numpy as np
from PIL import Image
import pickle



#####INPUT######
Pickle_file = "Test"
################

path=f"Dataset"
lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()

def getimagewithid(path):
    current_id = 0
    label_ids = {}
    faces=[]
    ids=[]
    for folderData in os.listdir(path):
        folder_path = f"Dataset/{folderData}"
        for imagepath in os.listdir(folder_path):
            facesim = Image.open(f"{folder_path}/{imagepath}").convert("L")
            imgnp = np.array(facesim, "uint8")
            id = int(imagepath.split(".")[0].split("_")[1])
            name =str(imagepath.split(".")[0].split("_")[0])
            if not name in label_ids:
                label_ids[name] = current_id
                current_id+=1    
            id = label_ids[name]
            faces.append(imgnp)
            ids.append(id)
            cv2.waitKey(10)

    return np.array(ids),faces,label_ids
    
       
ids,faces,label_ids = getimagewithid(path)

with open(f"Yaml/{Pickle_file}.pickle", 'wb' ) as f:
    pickle.dump(label_ids,f)



lbph_recognizer.train(faces,ids)
lbph_recognizer.save(f"Yaml/{Pickle_file}.yml")


