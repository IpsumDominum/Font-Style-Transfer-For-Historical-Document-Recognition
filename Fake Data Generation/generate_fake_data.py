import sys
sys.path.append("scripts")
from prepare_dataset import get_segmentation_lines
from gen_samples import generate_sample
import os
import pickle


def generate_recognition_data(num_samples,mode="Train"):
    if(mode=="Train"):
        save_dir = "samples"
        save_seg_dir = "segmented"
        label_filename = "train_labels.pkl"
        seg_label_filename = "train_seg_labels.pkl"
    else:
        save_dir = "samples_test_new"
        save_seg_dir = "segmented_test_new"
        label_filename = "test_labels.pkl"
        seg_label_filename = "test_seg_labels.pkl"
    labels = generate_sample(num_samples,save_dir)
    """
    Load saved labels
    """
    seg_labels = get_segmentation_lines(labels,save_dir,save_seg_dir)

    pickle.dump(labels,open(label_filename, 'wb'))
    pickle.dump(seg_labels,open(seg_label_filename, 'wb'))

if __name__=="__main__":
    generate_test(2,mode="Test")