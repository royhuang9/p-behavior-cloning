#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 21:02:31 2017

@author: roy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:11:37 2017

@author: roy
"""
from keras.models import model_from_json
import json

def load_model(file_name):
    print('load the mode:{}'.format(file_name))
    
    with open(file_name, 'r') as jfile:
        model = model_from_json(json.loads(jfile.read()))
  
    model.compile('adam', 'mse')
    
    weights_file = file_name.replace('json','h5')
    
    model.load_weights(weights_file)
    
    return model

model=load_model('./models/model_1221.json')

model.summary()