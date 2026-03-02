import os
import numpy as np
import random
import dask.dataframe as dd
from datasets.bases import BaseImageDataset
from utils.model_utils import is_main_process

class SewerMLZSMultiLabelClassification(BaseImageDataset):
    def __init__(self, root='', verbose=True, **kwargs):
        super(SewerMLZSMultiLabelClassification, self).__init__()
        self.dataset_dir = root
        dataset_name = "SewerML"
        self.train_file = os.path.join(self.dataset_dir, 'sewerml_train.csv')
        self.test_file = os.path.join(self.dataset_dir, 'sewerml_test.csv')
        self.test_file_gzsl = os.path.join(self.dataset_dir, 'sewerml_test.csv')

        self.seen_defect_classes = self._load_defects_from_file(type='seen')
        self.unseen_defect_classes = self._load_defects_from_file(type='unseen')
        self.defect_labels = self.seen_defect_classes + self.unseen_defect_classes 

        self._check_before_run()

        train, class2idx, name_train = self._load_dataset(self.dataset_dir, self.train_file, split='seen', shuffle=True)
        test, _, name_test = self._load_dataset(self.dataset_dir, self.test_file, split='unseen', shuffle=False)
        test_gzsl, _, _ = self._load_dataset(self.dataset_dir, self.test_file_gzsl, split='all', shuffle=False)
        self.train = train
        self.test = test
        self.test_gzsl = test_gzsl
        self.class2idx = class2idx
        if verbose and is_main_process():
            print(f"=> {dataset_name} ZSL Dataset:")
            self.print_dataset_statistics(train, test)
            print(f"=> {dataset_name} GZSL Dataset:")
            self.print_dataset_statistics(train, test_gzsl)
        self.classnames_seen = name_train
        self.classnames_unseen = name_test
        self.classnames = name_train + name_test
        self.num_cls_train = len(name_train)
        self.num_cls_test = len(name_test)

    def _load_defects_from_file(self, type='seen'):
        if type == 'seen':
            file_path = os.path.join('datasets/classes/sewerml', 'seen.txt')
        elif type == 'unseen':
            file_path = os.path.join('datasets/classes/sewerml', 'unseen.txt')
        else:
            raise ValueError("Invalid type. Expected 'seen' or 'unseen'.")
        
        with open(file_path, 'r') as f:
            defects = [line.strip() for line in f if line.strip()]
        
        return defects

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_file):
            raise RuntimeError("'{}' is not available".format(self.train_file))
        if not os.path.exists(self.test_file):
            raise RuntimeError("'{}' is not available".format(self.test_file))

    def _load_dataset(self, data_dir, annot_path, split, shuffle=True, names=None):
        out_data = []
        with open(annot_path) as f:
            if split == 'seen':
                classes = self.seen_defect_classes
            elif split == 'unseen':
                classes = self.unseen_defect_classes
            else:
                classes = self.defect_labels

            annotation = dd.read_csv(annot_path, sep=',', usecols= classes + ["Filename"])
            annotation = annotation.compute() 
            annotation = annotation.dropna(subset=['Filename'])

            class_to_idx = {classes[i]: i for i in range(len(classes))}
            images_paths = annotation['Filename'].values
            labels_matrix = annotation[classes].values 

            for i in range(len(images_paths)):
                img_path = images_paths[i]
                if not isinstance(img_path, str) or img_path.strip() == "":
                    continue
                    
                full_image_path = os.path.join(data_dir, 'data', img_path)
                labels = labels_matrix[i]
                
                out_data.append((full_image_path, labels.astype(np.float32)))

        if shuffle:
            random.shuffle(out_data)
        return out_data, class_to_idx, classes



