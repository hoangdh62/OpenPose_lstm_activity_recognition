import xml.etree.ElementTree as ET
from imutils import paths
import os

file1 = open("trainlist01.txt", "a", encoding="utf-8")
xmlPaths = list(paths.list_files('mydata/videos_train'))
for xmlPath in xmlPaths:
    label = xmlPath.split(os.path.sep)[-2]
    basename = os.path.basename(xmlPath)
    # class_id = ''
    # box_coordinates = ''
    # tree = ET.parse(xmlPath)
    # filename = tree.find('filename').text
    # # parse an xml file by name
    # objects = tree.getiterator('object')
    # for object in objects:
    #     name = object.find('name').text
    #     if name == "D00":
    #         class_id = '0'
    if label == "Eat":
        class_id = '1'
    elif label == "Sit":
        class_id = '2'
    elif label == "Sleep":
        class_id = '3'
    elif label == "Stand":
        class_id = '4'
    elif label == "Walk":
        class_id = '5'
    #     bndboxs = object.getiterator('bndbox')
    #     ymin, xmin, ymax, xmax = None, None, None, None
    #     for bndbox in bndboxs:
    #         xmin = bndbox.find("xmin").text
    #         ymin = bndbox.find("ymin").text
    #         xmax = bndbox.find("xmax").text
    #         ymax = bndbox.find("ymax").text
    #     box_coordinates += xmin + "," + ymin + "," + xmax + "," + ymax + "," + class_id + " "
    # if box_coordinates != '':
    file1.write(label + "/" + basename+" "+ class_id + "\n")
file1.close()