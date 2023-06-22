from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
import base64
import cv2
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] == "secret!"
socketio = SocketIO(app)


@app.route('/camera')
def camera():
    return render_template('camera.html')


def base64_to_image(base64_string):

    base64_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()
load_data = torch.load('data.pt')
embedding_list = load_data[0]
name_list = load_data[1]


@socketio.on("image")
def receive_image(image):

    image = base64_to_image(image)
    img = Image.fromarray(image)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                dist_list = []  # list of matched distances, minimum distance is used to identify the person

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list)  # get minumum dist value
                min_dist_idx = dist_list.index(
                    min_dist)  # get minumum dist index
                # get name corrosponding to minimum dist
                name = name_list[min_dist_idx]
                box = boxes[i]

                if min_dist < 0.90:

                    image = cv2.putText(image, name+' '+str(round(min_dist, 2)), (int(box[0]), int(
                        box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                image = cv2.rectangle(image, (int(box[0]), int(
                    box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)

                frame_resized = cv2.resize(image, (640, 360))
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                result, frame_encoded = cv2.imencode(
                    ".jpg", frame_resized, encode_param)

                processed_img_data = base64.b64encode(frame_encoded).decode()
                b64_src = "data:image/jpg;base64,"
                processed_img_data = b64_src + processed_img_data
                emit("processed_image", processed_img_data)

                # cv2.imencode encode image to streaming data
                # base64.b64encode encode string into the binary form.
                # .decode() Decode the bytes using the codec registered for encoding.
                #  "data:image/jpg;base64," prefix to the encoded base 64


if __name__ == "__main__":

    app.debug = True
    socketio.run(app)
