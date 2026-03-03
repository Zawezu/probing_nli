import os
from torch.utils.data import Dataset, DataLoader

DATA_FOLDER = "data"
SICK_EN_FOLDER = "sick_en"
SICK_EN_FILE = "SICK_annotated.txt"


sick_filepaths = {"en": f"./{DATA_FOLDER}/{SICK_EN_FOLDER}/{SICK_EN_FILE}",
                  "es": "unavailable"}

class SICKDataset(Dataset):
    def __init__(self, language, split):
        self.sentence_pairs = {}
        self.labels = {}
        filepath = sick_filepaths[language]

        self.load_sick_dataset(filepath, split)


    def load_sick_dataset(self, filepath, split):        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            next(f) # Skip first line, since it is the column names
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                data = [s.strip() for s in line.split("\t")]
                
                if data[-1].lower() != split:
                    continue
                
                pair_ID = int(data[0])
                # pair_type = data[1]
                sentence_A = data[2]
                # sentence_A_expRule = data[3]
                sentence_B = data[4]
                # sentence_B_expRule = data[5]
                # relatedness_score = float(data[6])
                label = data[7].lower()
                # entailment_AB = data[8]
                # entailment_BA = data[9]
                # sentence_A_original = data[10]
                # sentence_B_original = data[11]
                # sentence_A_dataset = data[12]
                # sentence_B_datase = data[13]
                label = data[7].lower()
                
                self.sentence_pairs[pair_ID] = (sentence_A, sentence_B)
                self.labels[pair_ID] = label
            
    def __getitem__(self, index):
        return self.sentence_pairs[index], self.labels[index]
    

