import csv
import os
import random
from .input_data_structure import InputExample


class TripletProcessor(object):
    """
    Reads in the a Triplet Dataset: Each line contains (at least) 3 columns, one anchor column (s1),
    one positive example (s2) and one negative example (s3)
    """

    def __init__(self, dataset_folder, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, has_header=False, delimiter="\t",
                 quoting=csv.QUOTE_NONE):
        self.dataset_folder = dataset_folder
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.s3_col_idx = s3_col_idx
        self.has_header = has_header
        self.delimiter = delimiter
        self.quoting = quoting

    def get_examples(self, filenames, max_examples=0, max_length=64):
        examples = []
        for filename in filenames:
            data = csv.reader(open(os.path.join(self.dataset_folder, filename), encoding="utf-8"),
                              delimiter=self.delimiter,
                              quoting=self.quoting)
            sens = []
            if self.has_header:
                next(data)
            for id, row in enumerate(data):
                s1 = row[self.s1_col_idx]
                s2 = row[self.s2_col_idx]
                s3 = row[self.s3_col_idx]
                sens.extend((s1, s2, s3))
            sens = list(set(sens))
            random.shuffle(sens)
            for id, sen in enumerate(sens):
                if len(sen) < max_length:
                    examples.append(InputExample(guid=filename + str(id), text_a=sen, label=1))

        if max_examples > 0 and len(examples) >= max_examples:
            examples = random.sample(examples, max_examples)

        return examples

    def get_labels(self):
        return [0, 1]
