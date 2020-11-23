Directory structure

codes
├── data_augmentation
│   └── adience
│   	└── faces
│   	└── folds
│   	└── results
│   └── images
│   └── mask.py
│   └── mask_adience_data.py
├── storage
│   └── checkpoints
│   └── results
├── InceptionV3_train.ipynb
├── ResNet50V2_train.ipynb


data_augmentation/adience <- stores original and augmented images (removed to save file space)

data_augmentation/images <- stores mask variations

storage/checkpoints <- stores model weights for epoch which achieved best accuracy (removed to save file space)

storage/results <- contains generated graphs for each model