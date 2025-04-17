import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os 
import tensorflow as tf


Annotations = "annotations"
Images = "images"



def read_annotations(path):
    tree = ET.parse(path)
    root = tree.getroot()

    objects = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        objects.append({
            'label': label,
            'bbox': [xmin, ymin, xmax, ymax]
        })

    return objects


def prep_images_and_save():
    saved_images = 0

    for annotation in os.listdir(Annotations):
        annotation_path = os.path.join(Annotations , annotation)
        objects = read_annotations(annotation_path)

        image_path = os.path.join(Images, annotation.replace('.xml', '.png'))
        

        for object in objects:
            label = object['label']
            bbox = object['bbox']


            image = cv2.imread(image_path)

            x1, y1, x2, y2 = bbox
            cropped_image = image[y1:y2, x1:x2]
            cropped_image = cv2.resize(cropped_image,(224,224))

            cv2.imwrite(f"saved_faces/{label}/{saved_images}.png", cropped_image)
            saved_images += 1
    print(f"Saved {saved_images} images.")


def prep_dataset(path):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse'
    )

    return train_generator











    

    

