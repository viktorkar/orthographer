import argparse
import os
import tkinter
from application import Application

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def main(args):
    Application(tkinter.Tk(), "Ortographer", args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--td_model', type=str, default="yolo" ,choices=["yolo", "east"],  help='Text detection model use', )
    parser.add_argument('-r', '--tr_model', type=str, default="dtrb", choices=["resnet", "dtrb"], help="Text recognition model to use.")
    parser.add_argument('-p', '--post_process', type=str, default="symspell" ,choices=["symspell", "trocr", "None"],  help='Improve text recognition by postprocessing')
    parser.add_argument('-s', '--spell_check', type=str, default="symspell", choices=["symspell"], help="Spellcheck method to use.")
    parser.add_argument('-v', '--video_source', type=int, default=0, choices=[0, 1], help='Input video source, 0=webcam, 1=connected device')
    args = parser.parse_args()

    print("""
        ################ Orthographer #########################
        Running program with the following parameters:
        Detection:      {}
        Recognition:    {}
        Postprocessing: {}
        Spell checker:  {}
        Video Source:   {}
    """.format(args.td_model, args.tr_model, args.post_process, args.spell_check, args.video_source))

    main(args)