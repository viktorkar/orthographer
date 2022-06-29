import argparse
import math
import os
import numpy as np
import pandas as pd
import cv2 as cv

from ocr import OCR
from symspellpy import editdistance
from tqdm import tqdm
from util import convert_wordlist_to_string

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SystemBenchmark:
    def __init__(self, args):
        self.args = args
        self.algorithm = editdistance.DistanceAlgorithm(2)
        self.ed = editdistance.EditDistance(self.algorithm)
        self.ocr = OCR(args, 32, 100, 'en')
        self.tickmeter = cv.TickMeter()

    ###########################################################################
    def __load_data(self):
        """Loads the data at the specified data_path. Returns a
        pandas dataframe containing rows of img_path, transcript and classes.
        Class = 0 if correct word, 1 if incorrect."""
        # Paths to all our data.
        images_dir = os.path.join(self.args.data_path, 'images')
        labels_dir = os.path.join(self.args.data_path, 'labels')

        data = {
        'img_path': [], 
        'transcript' : [],
        'classes' : [],
        }

        images = os.listdir(images_dir)

        for img in images:
            img_path = os.path.join(images_dir, img)
            label = img.split(".")[0] + ".txt"
            label_path = os.path.join(labels_dir, label)
            
            label_file = open(label_path, 'r')
            lines = label_file.readlines()
            transcript = []
            targets = []

            for line in lines:
                word, target = line.split(" ")
                target = int(target)
                transcript.append(word)
                targets.append(target)
            
            data['img_path'].append(img_path)
            data['transcript'].append(transcript)
            data['classes'].append(targets)
        
        return pd.DataFrame(data)

    ###########################################################################
    def __get_image(self, image_path):
        return cv.imread(image_path)

    ###########################################################################
    def __compare_strings(self, transcript, pred):
        """Compares two given strings using the levenshtein fast algorithm."""
        longest_string_length = max(len(transcript), len(pred))
        diff = self.ed.compare(pred, transcript, longest_string_length)
        error_rate = diff / longest_string_length
        return error_rate
    
    ###########################################################################
    def __calculate_spellcheck_metrics(self, transcript, rec_pred, classes_true, classes_pred):
        """Calculates the recall, precision and f1_score of the spellcheck predicitons."""
        # Extract the predicted incorrect words.
        recognized_words = np.array(rec_pred)
        classes_pred = np.array(classes_pred)
        pred_incorrect_words = set(recognized_words[classes_pred == 1])
        
        # Extract the true incorrect words.
        transcript = np.array(transcript)
        classes_true = np.array(classes_true)
        true_incorrect_words = set(transcript[classes_true == 1])

        # Calculate precision and recall.
        recall = float('nan')    # This way we can skip these values when calculating mean
        precision = float('nan') # if we can't calculate precision or recall for this example.
        f1 = float('nan')
        if (len(pred_incorrect_words) > 0):
            precision = len(pred_incorrect_words & true_incorrect_words) / len(pred_incorrect_words)
            precision = round(precision, 4)

        if (len(true_incorrect_words) > 0):
            recall = len(pred_incorrect_words & true_incorrect_words) / len(true_incorrect_words)
            recall = round(recall, 4)

        if (recall == 0 and precision == 0):
            f1 = 0
        elif not (math.isnan(precision) or math.isnan(recall)):
            f1 = 2 * (precision * recall) / (precision + recall)
            
        return precision, recall, f1

    ###########################################################################
    def run_benchmark(self):
        """Benchmarks the system. Returns a pandas dataframe containing all results."""
        benchmark_result = {
            "img_path" : [],
            "transcript" : [],
            "classes_true": [],
            "bounding_boxes" : [],
            "rec_pred" : [],
            "final_pred" : [],
            "rec_error" : [],
            "final_error" : [],
            "det_inf" : [],
            "rec_inf" : [],
            "pp_inf" : [],
            "sc_inf" : [],
            "classes_pred" : [],
            "suggestions" : [],
            "sc_precision" : [],
            "sc_recall" : [],
            "sc_f1" : [],
            "total_inf" : []
        }

        # Load the data.
        data = self.__load_data()

        if self.args.max_samples:
            data_size = min(self.args.max_samples, len(data))
            data = data.sample(n=data_size)

        # Loop through data and perform OCR.
        for i in tqdm(range(len(data)), desc="Progress"):
            x = data.iloc[i]
            img = self.__get_image(x.img_path)
            transcript = x.transcript
            classes_true = x.classes

            self.tickmeter.start()
            det_inf, rec_inf, pp_inf, sc_inf, boxes, rec_pred, final_pred, spellchecks = self.ocr.benchmark_ocr(img)
            self.tickmeter.stop()

            total_inf = self.tickmeter.getTimeMilli()
            self.tickmeter.reset()
            
            # Split spellchecks into classes and suggestions. (class=True => word incorrect)
            # We convert the booleans to ints.
            classes_pred, suggestions = zip(*spellchecks)
            classes_pred = list(map(int, classes_pred))
            suggestions = list(suggestions)
            
            # First convert the lists of words to one string.
            rec_pred_string = convert_wordlist_to_string(rec_pred)
            transcript_string = convert_wordlist_to_string(transcript)
            final_pred_string = convert_wordlist_to_string(final_pred)
            # Then compare the strings.
            rec_error = self.__compare_strings(transcript_string, rec_pred_string)
            final_error = self.__compare_strings(transcript_string, final_pred_string)

            # Calculate accuracy of spellcheck.
            sc_precision, sc_recall, sc_f1 = self.__calculate_spellcheck_metrics(transcript, final_pred, classes_true, classes_pred)

            # Divide inference times by number of detected words.
            nbr_words = len(boxes)
            if (nbr_words > 0):
                det_inf = round(det_inf / nbr_words, 2)
                rec_inf = round(rec_inf / nbr_words, 2)
                pp_inf = round(pp_inf / nbr_words, 2)
                sc_inf = round(sc_inf / nbr_words, 2)
                total_inf = round(total_inf / nbr_words, 2)

            benchmark_result["img_path"].append(x.img_path)
            benchmark_result['transcript'].append(transcript)
            benchmark_result['classes_true'].append(classes_true)
            benchmark_result['bounding_boxes'].append(boxes)
            benchmark_result['rec_pred'].append(rec_pred)
            benchmark_result['final_pred'].append(final_pred)
            benchmark_result['rec_error'].append(rec_error)
            benchmark_result['final_error'].append(final_error)
            benchmark_result['det_inf'].append(det_inf)
            benchmark_result['rec_inf'].append(rec_inf)
            benchmark_result['pp_inf'].append(pp_inf)
            benchmark_result['sc_inf'].append(sc_inf)
            benchmark_result['classes_pred'].append(classes_pred)
            benchmark_result['suggestions'].append(suggestions)
            benchmark_result['sc_precision'].append(sc_precision)
            benchmark_result['sc_recall'].append(sc_recall)
            benchmark_result['sc_f1'].append(sc_f1)
            benchmark_result['total_inf'].append(total_inf)
            
        return pd.DataFrame(benchmark_result)

###########################################################################
def evaluate_result(result):
    final_error = round(result['final_error'].mean() * 100, 3)
    rec_error = round(result['rec_error'].mean() * 100, 3)
    det_inf = round(result['det_inf'].mean(), 3)
    rec_inf = round(result['rec_inf'].mean(), 3)
    pp_inf = round(result['pp_inf'].mean(), 3)
    sc_inf = round(result['sc_inf'].mean(), 3)
    total_inf = round(result['total_inf'].mean(), 3)
    sc_precision = round(result['sc_precision'].mean() * 100, 3)
    sc_recall = round(result['sc_recall'].mean() * 100, 3)
    sc_f1 = round(result['sc_f1'].mean() * 100, 3)
    
    # Check if directory benchmark-results exists. If not, create it.
    if not os.path.exists("../benchmark-results"):
        os.mkdir("../benchmark-results")

    result.to_csv("../benchmark-results/" + args.output_name, sep=';', index=False)

    print("""
          -----------------------------------------------\n
          Original recognition error:      {} %
          Final recognition error:         {} %\n
          -----------------------------------------------\n
          Detection inference time:        {} ms / word
          Recognition inference time:      {} ms / word
          Postprocess inference time:      {} ms / word
          Spell correction inference time: {} ms / word
          Total inference time:            {} ms / word\n
          -----------------------------------------------\n
          Spellcheck precision: {} %
          Spellcheck recall:    {} %
          Spellcheck F1:        {} %\n
          -----------------------------------------------\n
          Results stored in: ../benchmark-results/{}
    """.format(rec_error, final_error, det_inf, rec_inf, pp_inf, sc_inf, total_inf, sc_precision, sc_recall, sc_f1, args.output_name))

###########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--td_model', type=str, default="yolo" ,choices=["yolo", "east"],  help='Text detection model use', )
    parser.add_argument('-r', '--tr_model', type=str, default="dtrb", choices=["resnet", "dtrb"], help="Text recognition model to use.")
    parser.add_argument('-p', '--post_process', type=str, default="symspell" ,choices=["symspell", "trocr", "None"],  help='Improve text recognition by postprocessing')
    parser.add_argument('-s', '--spell_check', type=str, default="symspell", choices=["symspell"], help="Spellcheck method to use.")
    parser.add_argument('-dp', '--data_path', type=str, default='../data/system_dataset/', help='Path to data folder')
    parser.add_argument('-m', '--max_samples', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('-o', '--output_name', type=str, default='benchmark-result.csv', help='Name of result file')
    args = parser.parse_args()

    print("""
        ################ BENCHMARK #########################
        Running benchmark with the following parameters:
        Detection:      {}
        Recognition:    {}
        Postprocessing: {}
        Spell checker:  {}
        Data path:      {}
        Max samples:    {}
        Output name:    {}
    """.format(args.td_model, args.tr_model, args.post_process, args.spell_check, args.data_path, args.max_samples, args.output_name))

    benchmark = SystemBenchmark(args)
    result = benchmark.run_benchmark()
    evaluate_result(result)


