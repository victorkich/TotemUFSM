from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from facedetector import FaceDetector
from maskdetector import MaskDetector
import numpy as np
import torch
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch CUDA available: {}".format(torch.cuda.is_available()))


def load_models():
    print('Carregando modelos...')
    mask_detector = MaskDetector()
    mask_detector.load_state_dict(torch.load('models/face_mask.ckpt')['state_dict'], strict=False)
    mask_detector = mask_detector.to(device)
    mask_detector.eval()

    face_detector = FaceDetector(
        prototype='models/deploy.prototxt.txt',
        model='models/res10_300x300_ssd_iter_140000.caffemodel',
    )
    return face_detector, mask_detector


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    image = cv2.UMat(image)
    return cv2.warpAffine(image, M, (nW, nH))


def draw_frame(image, c):
    ws = image.shape[0]
    hs = image.shape[1]
    image = cv2.UMat(image)
    image = cv2.line(image, (int(hs * 0.05), int(ws * 0.05)), (int(hs * 0.05), int(ws * 0.2)), c, 9)
    image = cv2.line(image, (int(hs * 0.05), int(ws * 0.05)), (int(hs * 0.2), int(ws * 0.05)), c, 9)
    image = cv2.line(image, (int(hs * 0.8), int(ws * 0.05)), (int(hs * 0.95), int(ws * 0.05)), c, 9)
    image = cv2.line(image, (int(hs * 0.95), int(ws * 0.05)), (int(hs * 0.95), int(ws * 0.2)), c, 9)
    image = cv2.line(image, (int(hs * 0.05), int(ws * 0.95)), (int(hs * 0.2), int(ws * 0.95)), c, 9)
    image = cv2.line(image, (int(hs * 0.05), int(ws * 0.8)), (int(hs * 0.05), int(ws * 0.95)), c, 9)
    image = cv2.line(image, (int(hs * 0.8), int(ws * 0.95)), (int(hs * 0.95), int(ws * 0.95)), c, 9)
    image = cv2.line(image, (int(hs * 0.95), int(ws * 0.8)), (int(hs * 0.95), int(ws * 0.95)), c, 9)
    return image


def just_show_frame(img):
    c = (10, 0, 255)
    img = draw_frame(img, c)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.UMat.get(img)
    cv2.imshow('UFSM', img)


if __name__ == '__main__':
    face_model, mask_model = load_models()
    video = cv2.VideoCapture(0)
    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])

    print('Iniciando...')
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ['Sem Mascara', 'Com Mascara']
    labelColor = [(255, 0, 10), (0, 255, 10)]
    predicted = 0
    while True:
        _, frame = video.read()
        frame = rotate_bound(frame, 270)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.UMat.get(frame)
        faces = face_model.detect(frame)
        big_w = 0
        big_h = 0
        if faces:
            for face in faces:
                _, _, w, h = face
                if big_h < h and big_w < w:
                    xStart, yStart, width, height = face

            # clamp coordinates that are outside of the image
            xStart, yStart = max(xStart, 0), max(yStart, 0)
            frame = cv2.UMat(frame)
            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (126, 65, 64),
                          thickness=2)
            frame = cv2.UMat.get(frame)
            w = frame.shape[0]
            h = frame.shape[1]

            if width > int(w * 0.35) and height > int(h * 0.35):
                if int(w * 0.9) > xStart+width and xStart > int(w * 0.1):
                    if int(h * 0.9) > yStart+height and yStart > int(h * 0.1):
                        faceImg = frame[yStart:yStart + height, xStart:xStart + width]
                        output = mask_model(transformations(faceImg).unsqueeze(0).to(device))
                        _, predicted = torch.max(output.data, 1)
                        frame = draw_frame(frame, labelColor[predicted])
                        cv2.putText(frame, labels[predicted], (int(w * 0.2), int(h * 0.1)), font, 1,
                                    labelColor[predicted], 2)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame = cv2.UMat.get(frame)
                        video.release()
                        cv2.imshow('UFSM', frame)
                        _ = cv2.waitKey(5000)
                        video = cv2.VideoCapture(0)
                    else:
                        just_show_frame(frame)
                else:
                    just_show_frame(frame)
            else:
                just_show_frame(frame)
        else:
            just_show_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    del mask_model
