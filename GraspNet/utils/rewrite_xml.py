#导入minidom
from xml.dom import minidom
from xml.etree.ElementTree import ElementTree,Element
import xml.etree.ElementTree as ET
import os
from os.path import join

def rewrite_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for i, elem in enumerate(root.iter('filename')):
        if elem.text[0:5]=='/home':
            if i==0:
                new_elem = elem.text[42:]
            elif i==1:
                new_elem = elem.text[17:]
            else:
                raise ValueError(".xml may error")
            elem.text = new_elem
            print("all changing done")
        else:
            print("nothing need change")
    tree.write(xml_path)



def rewrite_xmls(xml_path):
    for sub_name in sorted(os.listdir(xml_path)):
        if sub_name.endswith('.xml'):
            sub_path = join(xml_path, sub_name)
            print(sub_path)
            rewrite_xml(sub_path)

# if __name__ == "__main__":
#     xml_path = '/home/lm/graspit/worlds'
#     rewrite_xmls(xml_path)
#     print('==================----------rewrite .xml done!-----------======================')
