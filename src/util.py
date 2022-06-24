from collections import Counter

import math
import cv2 as cv
import numpy as np

######################################################################################################
def find_highest_prob_pred(predictions):
    """Finds the prediction with the highest probability."""
    highest_prob_idx = np.argmax(predictions, axis=0)[1]

    return predictions[highest_prob_idx]

######################################################################################################
def find_most_prev_pred(predictions):
    """Finds the most prevelent prediction in predictions."""
    c = Counter(pred[0] for pred in predictions) # predictions = [(pred_string, confidence), ...]
    best = c.most_common(1)[0][0]

    return best
    
######################################################################################################
def fourPointsTransform(frame, box):
    """Crops out a box from a frame given the 4 (x,y) corner points. Can handle rotated boxes."""
    vertices = [
        [box[0], box[3]],
        [box[0], box[2]],
        [box[1], box[2]],
        [box[1], box[3]]
    ]
    vertices = np.asarray(vertices, dtype='float32')
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv.getPerspectiveTransform(vertices, targetVertices)
    result = cv.warpPerspective(frame, rotationMatrix, outputSize)
    return result

######################################################################################################
def align_image(image):
    """Automatic image alignment using PCA components."""
    # Find positions [col, row] of all zero elements (text pixels).
    zero_positions = np.transpose(np.where(image==0)).astype('float')
    zero_positions = zero_positions[:, [1, 0]] # Change order from [row, col] to [col, row]
    N, _ = zero_positions.shape

    # Calculate and subtract respective mean. If mean for row > mean for col, 
    # do not perform alignemnt.
    mean = zero_positions.sum(axis=0) / N
    H, W = image.shape[:2]
    if mean[0] < mean[1] or W * 1.4 < H:
        return image

    zero_positions[:, 0] -= mean[0]
    zero_positions[:, 1] -= mean[1]

    # Calculate covariance matrix and eigenvectors.
    covariance = np.cov(zero_positions, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(covariance)

    # Construct A where first column is the eigenvector with largest eigenvalue.
    A = np.transpose(eig_vec)

    # Calculate rotation angle
    trA = np.matrix.trace(A)
    angle = np.degrees(np.arccos(trA / 2))

    # Determine right or left rotation.
    if angle > 90:
        angle = 180 - angle

    if eig_vec[0][0] * eig_vec[1][0] > 0:
        angle = -angle

    # Add white border to image to prevent black pixels at borders when performing rotation.
    image = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_CONSTANT, value = 255) 

    # Perform rotation
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    image_aligned = cv.warpAffine(image, M, (w, h), cv.INTER_CUBIC, cv.BORDER_CONSTANT, 1)

    return image_aligned

######################################################################################################
def concat_boxes_to_image(cropped, box_max_height, max_concats):
    """Concatenate cropped out words to images. Maximum max_concats number of words per image."""
    nbr_images = len(cropped) # The number of cropped images.
    nbr_concatenated = math.ceil(nbr_images / max_concats) # Number of resulting concatenated images.
    img_index = 0
    # Create x number of concatenated images.
    concatenated_images = []
    lengths = []
    for i in range(nbr_concatenated):
        cropped_padded = []

        for j in range(max_concats):
            # If all images has been padded and concatenated. Break
            if img_index >= nbr_images:
                break

            # Calculate padding based on box_max_height.
            crop = cropped[img_index]
            (height, width) = crop.shape[:2]
            r_var = 0
            v_padding = math.floor((box_max_height - height) / 2) # Vertical padding size
                # Due to rounding, padding can't always be equal on both sides of the image.
            if height + (2 * v_padding) < box_max_height:
                r_var = 1

            #print('Padding: ', v_padding, ' Height: ', line_height, ' new height of image: ', line_height + (2*v_padding) + r_var, ' max height: ', max_height)
            crop = cv.copyMakeBorder(crop, v_padding, v_padding+r_var, 0, 0, cv.BORDER_CONSTANT, value=255)
            cropped_padded.append(crop)
            img_index += 1

        concat_image = cv.hconcat(cropped_padded)
        concatenated_images.append(concat_image)
        lengths.append(len(cropped_padded))

    return concatenated_images, lengths

######################################################################################################
def crop_images(frame, boxes):
    """Crops out all boxes from a frame and performs some processign to each crop."""
    cropped = []
    box_max_height = 0

    # Crop out all images and find max height of box
    for box in boxes:
        # Crop image and perform some processing.
        crop = frame[box[2]:box[3], box[0]:box[1]]
        crop = cv.GaussianBlur(crop, (5,5), 0)
        _, crop = cv.threshold(crop, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        #crop = align_image(crop)

        # Add crop to cropped images.
        cropped.append(crop)

        # Check if crop height > max_height
        crop_height = crop.shape[0]
        if crop_height > box_max_height:
            box_max_height = crop_height
    
    return cropped, box_max_height

######################################################################################################
def convert_wordlist_to_string(list_of_words):
    """Takes a list of single words and returns the full transcript."""
    nbr_words = len(list_of_words)
    string = ""

    if nbr_words > 0:
        string += list_of_words[0]
        
        for i in range(1, nbr_words):
            string += (" " + list_of_words[i])

    return string