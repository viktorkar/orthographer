"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import Counter
import math
import cv2 as cv
import torch
import torch.nn as nn

from dtrb.modules.transformation import TPS_SpatialTransformerNetwork
from dtrb.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor, DenseNet
from dtrb.modules.sequence_modeling import BidirectionalLSTM
from dtrb.modules.prediction import Attention
import torchvision.transforms.functional as TF
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):

    def __init__(self, opt, converter):
        super(Model, self).__init__()
        self.opt = opt
        self.converter = converter
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'DenseNet':
            self.FeatureExtraction = DenseNet(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        
        if opt.FeatureExtraction == 'DenseNet':
            self.FeatureExtraction_output = 768 # DenseNet output channel is 768
        else:
            self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        # self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    #############################################################################
    def __resize_image(self, image):
        height, width = image.shape[:2]
        padH, padW = False, False

        if height < self.opt.imgH:
            padH = True
            padding = math.floor((self.opt.imgH - height)/2)
            rest_var = 0
            if height + (2 * padding) < self.opt.imgH: # Due to rounding we might have to add one to one side of the padding
                rest_var = 1
            image = cv.copyMakeBorder(image,padding,padding+rest_var,0,0,cv.BORDER_CONSTANT, value=255)
        if width < self.opt.imgW:
            padW = True
            padding = math.floor((self.opt.imgW - width)/2)
            rest_var = 0
            if width + (2 * padding) < self.opt.imgW: # Due to rounding we might have to add one to one side of the padding
                rest_var = 1
            image = cv.copyMakeBorder(image,0,0,padding,padding+rest_var,cv.BORDER_CONSTANT, value=255)
        if not (padH and padW):
            image = cv.resize(image, (self.opt.imgW, self.opt.imgH))

        return image

    #############################################################################
    def forward(self, cropped, text=None, is_train=False):
        cropped = self.__resize_image(cropped)
        input = TF.to_tensor(cropped)
        input.unsqueeze_(0)
        input = input.to(device)

        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = visual_feature.permute(0, 3, 1, 2)
        # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        # """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * 1).to(device)
        text_for_pred = torch.LongTensor(1, self.opt.batch_max_length + 1).fill_(0).to(device)
        b, w, c = contextual_feature.size()
        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            contextual_feature = contextual_feature.view(b*w, c)
            prediction = self.Prediction(contextual_feature)
            prediction = prediction.view(b,w,-1) 
            preds_prob = F.softmax(prediction, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            preds_size = torch.IntTensor([prediction.size(1)] * 1)
            _, preds_index = prediction.max(2)

            preds_max_prob, _ = preds_prob.max(dim=2)
            confidence_score = preds_max_prob[0].cumprod(dim=0)[-1]

            prediction = self.converter.decode(preds_index.data, preds_size.data)[0]
        else:
            # Note that this OpenCV DNN version dose not support Transformer.
            prediction = self.Prediction(contextual_feature.contiguous(), text_for_pred, is_train=False)
            prediction = prediction.view(b,w,-1) 
            preds_prob = F.softmax(prediction, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            _, preds_index = prediction.max(2)

            preds_max_prob, _ = preds_prob.max(dim=2)
            prediction = self.converter.decode(preds_index, length_for_pred)[0]
            pred_EOS = prediction.find('[s]')
            prediction = prediction[:pred_EOS]  # prune after "end of sentence" token ([s])

            pred_max_prob = preds_max_prob[0][:pred_EOS]
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

        return prediction, confidence_score.item()