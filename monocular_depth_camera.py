import cv2 as cv
#download model di : https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx
model = cv.dnn.readNet("model-small.onnx") #direktori model
model.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h,w,c = img.shape
    blob = cv.dnn.blobFromImage(img, 1/255.0, (256,256), (123.675, 116.28, 103.53), True, False)
    model.setInput(blob)
    output = model.forward()
    output = output[0,:,:]
    output = cv.resize(output, (w,h))
    output = cv.normalize(output, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imshow('img', img)
    cv.imshow('output', output)
    if cv.waitKey(10) & 0xFF == 27:
        break
cap.release()
cv.destroyAllWindows()
