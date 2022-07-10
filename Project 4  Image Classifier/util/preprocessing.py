import numpy as np
import tensorflow as tf

def process_image(image):
    image= np.squeeze(image)
    resized_image = tf.image.resize(image, (224, 224))/255.0
    #resized_image =np.reshape(resized_image,(224,224,3))
    return resized_image
    
    
def predict(image_path,model,top_k):    
    prediction = model.predict(np.expand_dims(image_path, axis=0))
    result_values, result_indices = tf.math.top_k(prediction,top_k)
    classes = [class_names[str(element + 1)] for element in result_indices.numpy()[0]]
    return result_values.numpy()[0], classes