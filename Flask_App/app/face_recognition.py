import numpy as np
import sklearn
import pickle
import cv2
import os

# ============================
# FIX ĐƯỜNG DẪN THEO PROJECT ROOT
# ============================

# Lấy đường dẫn gốc của project (folder chứa thư mục app/, model/, templates/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "model")

haar_path = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
svm_path  = os.path.join(MODEL_DIR, "model_svm.pickle")
pca_path  = os.path.join(MODEL_DIR, "pca_dict.pickle")

# Load model
haar = cv2.CascadeClassifier(haar_path)
model_svm = pickle.load(open(svm_path, mode='rb'))
pca_models = pickle.load(open(pca_path, mode='rb'))

model_pca = pca_models['pca']
mean_face_arr = pca_models['mean_face']



def faceRecognitionPipeline(filename, path = True):
    if path:  
        #step1: read img
        img = cv2.imread(filename)
    else:
        img = filename
    #step2: convert into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #step3: crop the face
    faces = haar.detectMultiScale(gray, 1.5, 3)
    
    predictions = []
    
    for x,y,w,h in faces:
        
        roi = gray[y:y+h, x:x+w]
    
        #step4: normalization
        roi = roi / 255.0
        #step5: resize images (300, 300)
        if roi.shape[1] > 300:
            roi_resize = cv2.resize(roi, (300, 300), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (300, 300), cv2.INTER_CUBIC)
    
        #step6: flattening
        roi_reshape = roi_resize.reshape(1, 90000)
        #step7: subtract with mean
        roi_mean = roi_reshape - mean_face_arr 
        #step8: get eigen img (apply roi_mean to pca)
        eigen_image = model_pca.transform(roi_mean)
        #step9: eigen img for visualization
        eig_img = model_pca.inverse_transform(eigen_image)
        #step10: pass to ml model
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
    
        #step 11: generate report
        text = "%s : %d"%(results[0], prob_score_max*100)
        print(text)
        #defining color base on results
        if results[0] == 'male':
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.rectangle(img, (x,y-30), (x+w, y), color, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        output = {
            'roi' : roi,
            'eig_img' : eig_img,
            'prediction_name' : results[0],
            'score': prob_score_max
        }
    
        predictions.append(output)

    return img, predictions
        
    