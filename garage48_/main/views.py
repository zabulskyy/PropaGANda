from django.shortcuts import render
from rest_framework.response import Response
from .classifier import *

import keras
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_usage
set_session(tf.Session(config=config))

from rest_framework.views import APIView

classifier = load_model(path_to_model)

def index(request):
    return render(request, 'index.html', {})


class Api_v_1(APIView):
    # parser_classes = (MultiPartParser, FormParser)
    # permission_classes = (permissions.AllowAny,)

    def post(self, request, *args, **kwargs):
        text = request.data['text']


        vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            path_to_vord2vec,
            binary=False)
        vec_model.init_sims(replace=True)



        return Response(
            "Fu** you with your text - " + analysis_api(classifier, text))
