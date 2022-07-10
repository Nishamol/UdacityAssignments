import argparse
from util import preprocessing as pp
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image


def predict(image_path,model,top_k):    
    prediction = model.predict(np.expand_dims(image_path, axis=0))
    result_values, result_indices = tf.math.top_k(prediction,top_k)
    return result_values.numpy()[0], result_indices.numpy()[0]

def main():
    class_names = None
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('saved_model')
    parser.add_argument('--top_k', type = int, action="store", dest = "top_k")
    parser.add_argument('--category_names',action = "store", dest = "category_names")
    args = parser.parse_args()
    print(args)
    top_k = 1
    if args.top_k is not None: top_k = args.top_k
    if args.category_names is not None: 
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    model = tf.keras.models.load_model(args.saved_model,compile = False, custom_objects={'KerasLayer':hub.KerasLayer})
    model.load_weights('best_model').expect_partial()	
    test_image = np.array(Image.open(args.path))
    processed_test_image = pp.process_image(test_image)
    prob, indices = predict(processed_test_image,model,top_k)
    if class_names is not None:
        classes = [class_names[str(element + 1)] for element in indices]
        print(list(zip(classes,prob)))
    else:
        print(list(zip(indices,prob)))
    
if __name__ == "__main__":
    main()
    

