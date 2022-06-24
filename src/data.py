import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import fromstring
from typing import Union
import html
import math
import re
import json
import cv2 as cv

class IAM:
    def __init__(self):
        self.data_path = Path("../data/IAM")

    ##############################################################################
    def read_xml(self, xml_file: Union[Path, str]) -> ET.Element:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return root

    ##############################################################################
    def find_child_by_tag(self, xml_el: ET.Element, tag: str, value: str) -> Union[ET.Element, None]:
        for child in xml_el:
            if child.get(tag) == value:
                return child

        return None

    ##############################################################################
    def find_word(self, xml_root: ET.Element, word_id: str, skip_bad_segmentation: bool = False) -> Union[str, None]:
        line_id = "-".join(word_id.split("-")[:-1])
        line = self.find_child_by_tag(xml_root[1].findall("line"), "id", line_id)

        if line is not None and not (skip_bad_segmentation and line.get("segmentation") == "err"):
            word = self.find_child_by_tag(line.findall("word"), "id", word_id)
            if word is not None:
                return html.unescape(word.get("text"))

        return None

    ##############################################################################
    def find_line(self, xml_root: ET.Element, line_id: str, skip_bad_segmentation: bool = False,) -> Union[str, None]:
        line = self.find_child_by_tag(xml_root[1].findall("line"), "id", line_id)

        if line is not None and not (
            skip_bad_segmentation and line.get("segmentation") == "err"):
            return html.unescape(line.get("text"))

        return None

    ##############################################################################
    def find_bounding_boxes(self, xml_line: ET.Element):
        """ Gets all bounding boxes of words in a line.

        Args:
            xml_line (ET.Element): line xml element

        Returns:
            A list of lists, containing box positions: x_start, x_end, y_start, y_end
            for every word.
        """
        boxes = []

        for word in xml_line.iter('word'):
            # Make sure we only get the bounding box if it is a word and not punctuation symbols.
            text = word.get('text')
            text = re.sub(r'[^\w]', '', text) 
            if text != '':
                cmps = word.findall('cmp')
                if cmps:
                    x_start = float(cmps[0].get('x'))
                    x_end = float(cmps[-1].get('x')) + float(cmps[-1].get('width'))
                    y_start = math.inf
                    y_end = 0

                    for cmp in cmps:
                        y_low = int(cmp.get('y'))
                        height = int(cmp.get('height'))
                        y_height = y_low + height

                        if y_height > y_end:
                            y_end = y_height

                        if y_low < y_start:
                            y_start = y_low

                    boxes.append([x_start, x_end, y_start, y_end])

        return boxes

    ##############################################################################       
    def get_words(self, skip_bad_segmentation: bool = False) -> pd.DataFrame:
        """Read all word images from the IAM dataset.

        Args:
            skip_bad_segmentation (bool): skip lines that have the
                segmentation='err' xml attribute
        Returns:
            List of 2-tuples, where each tuple contains the path to a word image
            along with its ground truth text.
        """

        data = {"img_path": [], "img_id": [], "target": []}
        root = self.data_path / "words"

        for d1 in root.iterdir():

            for d2 in d1.iterdir():
                doc_id = d2.name
                xml_root = self.read_xml(self.data_path / "xml" / (doc_id + ".xml"))

                for img_path in d2.iterdir():
                    target = self.find_word(xml_root, img_path.stem, skip_bad_segmentation)

                    if target is not None:
                        data["img_path"].append(str(img_path.resolve()))
                        data["img_id"].append(doc_id)
                        data["target"].append(target)

        return pd.DataFrame(data)

    ##############################################################################      
    def get_forms(self, skip_bad_segmentation = False) -> pd.DataFrame:
        """Read all form images from the IAM dataset.
        Args:
            skip_bad_segmentation (bool): skip bounding boxes of words present in lines
            with the line segmentation='err' xml attribute.

        Returns:
            pd.DataFrame
                A pandas dataframe containing the image path, image id, vertical
                upper bound, vertical lower bound, bounding box for every word, upper 
                and lower bound for every line and whether a line has segmentation errors.
        """
        data = {
            "img_path": [],
            "img_id": [],
            "bb_y_start": [],
            "bb_y_end": [],
            "bounding_boxes": [],
            "line_y0" : [],
            "line_y1" : [],
            "bad_segmentation": [],
        }
        root = self.data_path / "forms"
        for form_dir in ["formsA-D", "formsE-H", "formsI-Z"]:
            dr = root / form_dir

            for img_path in dr.iterdir():

                doc_id = img_path.stem
                xml_root = self.read_xml(self.data_path / "xml" / (doc_id + ".xml"))

                # Based on some empiricial evaluation, the 'asy' and 'dsy'
                # attributes of a line xml tag seem to correspond to its upper and
                # lower bound, respectively. We add padding of 10 pixels.
                bb_y_start = int(xml_root[1][0].get("asy")) - 10
                bb_y_end = int(xml_root[1][-1].get("dsy")) + 10

                bounding_boxes = []
                line_y0 = []
                line_y1 = []
                bad_segmentation = []

                for line in xml_root.iter("line"):
                    if line.get("segmentation") == "err":
                        bad_segmentation.append(True)

                        if not skip_bad_segmentation:
                            bounding_boxes.extend(self.find_bounding_boxes(line))
                    else:
                        bad_segmentation.append(False)
                        bounding_boxes.extend(self.find_bounding_boxes(line))

                    line_y0.append(int(line.get("asy")) - 10)
                    line_y1.append(int(line.get("dsy")) + 10)
                    
                data["img_path"].append(str(img_path))
                data["img_id"].append(doc_id)
                data["bb_y_start"].append(bb_y_start)
                data["bb_y_end"].append(bb_y_end)
                data["bounding_boxes"].append(bounding_boxes)
                data["line_y0"].append(line_y0)
                data["line_y1"].append(line_y1)
                data["bad_segmentation"].append(bad_segmentation)

        return pd.DataFrame(data)

    ############################################################################## 
    def get_lines(self, skip_bad_segmentation = False) -> pd.DataFrame:
        """Read all line images from the IAM dataset.

        Args:
            skip_bad_segmentation (bool): skip lines that have the
                segmentation='err' xml attribute
        Returns:
            List of 2-tuples, where each tuple contains the path to a line image
            along with its ground truth text.
        """

        data = {"img_path": [], "img_id": [], "target": []}

        root = self.data_path / "lines"
        for d1 in root.iterdir():
            for d2 in d1.iterdir():
                doc_id = d2.name
                xml_root = self.read_xml(self.data_path / "xml" / (doc_id + ".xml"))

                for img_path in d2.iterdir():
                    target = self.find_line(xml_root, img_path.stem, skip_bad_segmentation)

                    if target is not None:
                        data["img_path"].append(str(img_path.resolve()))
                        data["img_id"].append(doc_id)
                        data["target"].append(target)

        return pd.DataFrame(data)

###########################################################################################################
class IAM_yolo:
    def __init__(self):
        self.data_path = Path("../data/IAM_yolo")

    ##############################################################################      
    def get_data(self, split) -> pd.DataFrame:
        data = {
            "img_path": [],
            "img_id": [],
            "bounding_boxes": [],
        }
        root = self.data_path / "images" / split


        for img_path in root.iterdir():

            doc_id = img_path.stem
            target_path = self.data_path / "labels" / split / (doc_id + ".txt")
            img = cv.imread(str(img_path))
            img_height, img_width = img.shape[:2]
            
            bounding_boxes = []
            with open(target_path) as f:
                lines = f.read().splitlines() # Read all lines
                
            for line in lines:
                coordinates = line.split(' ')[1:]
                coordinates_float = [float(x) for x in coordinates] # Convert to float from string
                
                x_center, y_center, width, height = coordinates_float
                
                xmin = int((x_center - (width * 0.5)) * img_width)
                xmax = int((x_center + (width * 0.5)) * img_width)
                ymin = int((y_center - (height * 0.5)) * img_height)
                ymax = int((y_center + (height * 0.5)) * img_height)
                
                bounding_boxes.append([xmin, xmax, ymin, ymax])

            data["img_path"].append(str(img_path))
            data["img_id"].append(doc_id)
            data["bounding_boxes"].append(bounding_boxes)

        return pd.DataFrame(data)
        
###########################################################################################################
class CVL:
    def __init__(self):
        self.data_path = Path("../data/CVL")
        self.prefix_map = {"a": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"}

    ##############################################################################
    def read_xml(self, xml_file: Union[Path, str]) -> ET.Element:

        #open text file in read mode
        text_file = open(xml_file, "r")
        data_string = text_file.read()
        utf =  data_string.split("UTF-")[1][:2]
        if utf == "16":
            #encode data for utf-16
            data_string = data_string.encode('utf-16-be')
        elif utf == '8"':
            data_string = data_string.encode()
        et_tree = ET.ElementTree(fromstring(data_string))
        et_root = et_tree.getroot()

        return et_root

    ##############################################################################
    def words(self, xml_root: ET.Element):
        words = []
        for word in xml_root.findall(".//a:Page/a:AttrRegion/a:AttrRegion/a:AttrRegion/a:AttrRegion", self.prefix_map):
            bounding_box = []
            target = word.get("text")
            img_id = word.get("id")
            if target:
                for point in word.findall(".//a:Point", self.prefix_map):
                    bounding_box.append((point.get("x"), point.get("y")))
                #print("Word: ", word.get("text"), " with bounding box: ", bounding_box)
                words.append( (target, img_id, [bounding_box[0][0], bounding_box[0][1], bounding_box[2][0], bounding_box[2][1]]) ) # Append target, x_min, y_min, x_max, y_max
        return words

    ##############################################################################
    def boxes(self, xml_root: ET.Element):
        """Returns all bounding boxes, document_id and start of bounding boxes of a form in the CVL database.
        Returns:
            Tuple, containing the doc id of form, along with start of bounding boxes (in vertical direction),
            and a list of bounding boxes (in 'x_min, y_min, x_max, y_max' format)
        """
        boxes = []
        doc_id = ""
        bb_y_start = ""
        for page in xml_root.findall(".//a:Page", self.prefix_map):
            doc_id = page.get("imageFilename")
            doc_id = doc_id[:-4]
        for dkTL in page.findall(".//a:dkTextLines", self.prefix_map):
            bb_y_start = dkTL.get("cropY")
        for word in xml_root.findall(".//a:Page/a:AttrRegion/a:AttrRegion/a:AttrRegion/a:AttrRegion", self.prefix_map):
            target = word.get("text")
            if target:
                bounding_box = []
                for point in word.findall(".//a:Point", self.prefix_map):
                    bounding_box.append((point.get("x"), point.get("y")))
                #print("Word: ", word.get("text"), " with bounding box: ", bounding_box)
                boxes.append([bounding_box[0][0], bounding_box[0][1], bounding_box[2][0], bounding_box[2][1]])
                # append: x_min, y_min, x_max, y_max
        return (doc_id, bb_y_start, boxes)

    ##############################################################################
    def get_words(self) -> pd.DataFrame:
        """Read all word images from the IAM dataset.
        Returns:
            pd.Dataframe, where each row contains the path to a word image
            along with its ground truth text.
        """
        data = {"img_path": [], "img_id": [], "target": []}
        root = self.data_path

        for d1 in root.iterdir():
            d1 = d1 / "xml"
            for d2 in d1.iterdir():
                xml_root = self.read_xml(d2)
                words = self.words(xml_root)

                for (target, img_id, [x_min, y_min, x_max, y_max]) in words:
                    img_path = str(d1)
                    img_path = img_path[0:len(img_path) - 3]
                    img_path = img_path + "words\\" + img_id[:4] + "\\" + img_id + "-" + target + ".tif"
                    data["img_path"].append(img_path)
                    data["img_id"].append(img_id)
                    data["target"].append(target)

        return pd.DataFrame(data)

    ##############################################################################
    def get_forms(self) -> pd.DataFrame:
        """Read all form images from the IAM dataset.
        Returns:
            pd.DataFrame
                A pandas dataframe containing the image path, image id, target, vertical
                upper bound, vertical lower bound, target length, bounding box for every word.
        """
        data = {
            "img_path": [],
            "img_id": [],
            "bb_y_start": [],
            "bounding_boxes": [],
        }
        root = self.data_path
        for d1 in root.iterdir():
            d1 = d1 / "xml"
            for d2 in d1.iterdir():
                xml_root = self.read_xml(d2)
                (doc_id, bb_y_start, boxes) = self.boxes(xml_root)
                if doc_id:
                    img_path = str(d1)
                    img_path = img_path[0:len(img_path) - 3]
                    img_path = img_path + "pages\\" + doc_id + ".tif"

                    data["img_path"].append(str(img_path))
                    data["img_id"].append(doc_id)
                    data["bb_y_start"].append(bb_y_start)
                    data["bounding_boxes"].append(boxes)

        return pd.DataFrame(data)

    ##############################################################################
    def get_lines(self, skip_bad_segmentation = False) -> pd.DataFrame:
        """Read all line images from the IAM dataset.

        Args:
            skip_bad_segmentation (bool): skip lines that have the
                segmentation='err' xml attribute
        Returns:
            List of 2-tuples, where each tuple contains the path to a line image
            along with its ground truth text.
        """

        data = {"img_path": [], "img_id": [], "target": []}

        root = self.data_path / "lines"
        for d1 in root.iterdir():
            for d2 in d1.iterdir():
                doc_id = d2.name
                xml_root = self.read_xml(self.data_path / "xml" / (doc_id + ".xml"))

                for img_path in d2.iterdir():
                    target = self.find_line(xml_root, img_path.stem, skip_bad_segmentation)

                    if target is not None:
                        data["img_path"].append(str(img_path.resolve()))
                        data["img_id"].append(doc_id)
                        data["target"].append(target)

        return pd.DataFrame(data)

###########################################################################################################
class Imgur5k:
    def __init__(self):
        self.root_path = "../data/IMGUR5K/"
        self.info_path = self.root_path + "dataset_info/"
        self.data_path = self.root_path + "data/"

    ##############################################################################
    def get_dict(self):
        result = {}
        annotation_file = self.info_path + "imgur5k_annotations.json"
        with open(annotation_file) as json_file:
            json_data = json.load(json_file)
        image_ids = list(json_data['index_id'].keys())
        for image_id in image_ids:
            for word_id in json_data['index_to_ann_map'][image_id]:
                target = json_data['ann_id'][word_id]['word']
                box  = json_data['ann_id'][word_id]['bounding_box']
                if image_id not in result:
                    result[image_id] = {}
                    result[image_id]['image_path'] = '/data/' + image_id + ".jpg"
                    result[image_id]['targets'] = []
                    result[image_id]['bounding_boxes'] = []
                    result[image_id]['word_ids'] = []
                if 'targets' in result[image_id]:
                    result[image_id]['targets'].append(target)
                    result[image_id]['bounding_boxes'].append(box)
                    result[image_id]['word_ids'].append(word_id)
        return result

    ##############################################################################
    def get_words(self) -> pd.DataFrame:
        """Read all word images from the IAM dataset.
        Returns:
            pd.Dataframe, where each row contains the path to a word image
            along with its ground truth text.
        """
        data = {"img_path": [], "img_id": [], "target": [], "bounding_box": [], "orig_img_id": []}
        imgur_dict = self.get_dict()
        for (img_id, img_dict) in list(imgur_dict.items()):
            for i in range(len(img_dict['word_ids'])):
                word_id = img_dict['word_ids'][i]
                target = img_dict['targets'][i]
                box = img_dict['bounding_boxes'][i]
                img_path = self.root_path + 'words/images/' + word_id + '.jpg'
                data["img_path"].append(img_path)
                data["img_id"].append(word_id)
                data["target"].append(target)
                data["bounding_box"].append(box)
                data["orig_img_id"].append(img_id)
        return pd.DataFrame(data)

    ##############################################################################
    def get_splits(self):
        splits = ['train', 'val', 'test']
        result = {'train': [], 'val': [], 'test': []}
        for split in splits:
            split_ids = []
            split_file = self.info_path + split + "_index_ids.lst"
            f = open(split_file, "r")
            for img_id in f:
                split_ids.append(img_id[:-1])
            result[split] = split_ids
        return result