import os
from time import mktime
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request
import matplotlib.image as matimg

# BASE_DIR = thư mục 4_Flask_App
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# Thư mục upload
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'upload')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Thư mục lưu ảnh dự đoán
PREDICT_FOLDER = os.path.join(BASE_DIR, 'static', 'predict')
if not os.path.exists(PREDICT_FOLDER):
    os.makedirs(PREDICT_FOLDER)


def index():
    if request.method == 'POST':
        f = request.files.get('image_name')
        if f:
            # Lưu ảnh upload
            filename = f.filename
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(upload_path)
            print("Uploaded file saved at:", upload_path)

            # Xử lý dự đoán
            pred_image, predictions = faceRecognitionPipeline(upload_path)
            pred_filename = 'prediction_image.jpg'
            pred_path = os.path.join(PREDICT_FOLDER, pred_filename)
            cv2.imwrite(pred_path, pred_image)
            print("Prediction saved at:", pred_path)

            report = []
            for i, obj in enumerate(predictions):
                gray_image = obj['roi']
                eigen_image = obj['eig_img'].reshape(300, 300)
                gender_name = obj['prediction_name']
                score = round(obj['score'] * 100, 2)


                gray_image_name = f'roi_{i}.jpg'
                eigen_image_name = f'eigen_{i}.jpg'
                # dùng đường dẫn tuyệt đối
                gray_image_path = os.path.join(PREDICT_FOLDER, gray_image_name)
                eigen_image_path = os.path.join(PREDICT_FOLDER, eigen_image_name)

                matimg.imsave(gray_image_path, gray_image, cmap='gray')
                matimg.imsave(eigen_image_path, eigen_image, cmap='gray')


                report.append([gray_image_name,
                               eigen_image_name,
                               gender_name,
                               score])

            
            return render_template('base.html', fileupload=True, report=report)

    return render_template('base.html')
