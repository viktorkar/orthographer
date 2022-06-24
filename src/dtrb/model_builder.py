import torch
from dtrb.model import Model
from dtrb.utils import CTCLabelConverter, AttnLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DTRB_Opt:
    def __init__(self):
        self.workers = 0 # number of data loading workers, default=4
        self.saved_model = "../models/TR/orthonet_general.pth"
        """ Data processing """
        self.batch_max_length = 25 # maximum-label-length, default=25
        self.imgH = 32  # height of the input image, default=32 
        self.imgW = 100  # width of the input image, default=100
        self.rgb = False # use rgb input or not, default=True
        self.character = '0123456789abcdefghijklmnopqrstuvwxyz' # character used in the input data
        self.sensitive = False # case-sensitive or not
        self.PAD = False # whether to keep ratio then pad for image resize
        """ Model Architecture """
        self.Transformation = 'None'  # transformation stage. None | TPS
        self.FeatureExtraction = 'ResNet' # FeatureExtraction stage. VGG|RCNN|ResNet
        self.SequenceModeling = 'BiLSTM' # SequenceModeling stage. None|BiLSTM
        self.Prediction = 'CTC' # Prediction stage. CTC|Attn
        self.num_fiducial = 20 # number of fiducial points of TPS-STN
        self.input_channel = 1 # number of input channels of Feature extractor
        self.output_channel = 512 # number of output channels of Feature extractor
        self.hidden_size = 256 # number of hidden units of the LSTM
        self.num_gpu = torch.cuda.device_count()
        self.num_class = 0

def create_opt():
    opt = DTRB_Opt()

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)

    opt.num_class = len(converter.character)

    return opt, converter

def create_dtrb_model():
    opt, converter = create_opt()
    print('\nHTR MODEL: ', opt.saved_model,"\n")
    model = Model(opt, converter)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(opt.saved_model), strict=False)
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device), strict=False)

    model.eval()

    return model
