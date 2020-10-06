from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph


img_height, img_width=224,224
with open('./models/imagenet_classes.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)


model_graph = Graph()
with model_graph.as_default():
    gpuoptions = tf.compat.v1.GPUOptions(allow_growth=True)
    tf_session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpuoptions))
    with tf_session.as_default():
        model=load_model('./models/MobileNetModelImagenet.h5')



def index(request):
    context={'a':1}
    return render(request,'index.html',context)



def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x=x/255
    x=x.reshape(1,img_height, img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)

    import numpy as np
    #predictedLabel=labelInfo[str(np.argmax(predi[0]))]

    results = []
    for idx, x in np.ndenumerate(predi[0]):
        results.append ({"label": labelInfo[str(idx[0])][1], "prediction": x})
    sortedResults = sorted(results, key=lambda x: x["prediction"], reverse=True)


    #context={'filePathName':filePathName,'predictedLabel':predictedLabel[1]}
    context={'filePathName':filePathName,'predictedResults':sortedResults[:5]}
    return render(request,'index.html',context) 

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context) 