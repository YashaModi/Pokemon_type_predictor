import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional
from pokemon_predictor import config

class PokemonDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Data Generator for Pokemon Images.
    Handles loading, resizing, and augmentation.
    """
    def __init__(self, 
                 image_paths: List[str], 
                 labels: np.ndarray, 
                 batch_size: int = 32, 
                 dim: Tuple[int, int] = (224, 224), 
                 n_channels: int = 3,
                 shuffle: bool = True,
                 augment: bool = False):
        """
        Args:
            image_paths: List of absolute paths to images.
            labels: Metadata or One-Hot Encoded labels.
            batch_size: Batch size.
            dim: Image dimensions (H, W).
            n_channels: Number of channels (3 for RGB).
            shuffle: Whether to shuffle data at the end of each epoch.
            augment: Whether to apply data augmentation.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_paths_temp = [self.image_paths[k] for k in indexes]
        y = self.labels[indexes]

        # Generate data
        X = self.__data_generation(list_paths_temp)

        return X, y

    def __data_generation(self, list_paths_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, path in enumerate(list_paths_temp):
            # Load Image
            img = cv2.imread(path)
            if img is None:
                # Handle missing image (return blank or handled upstream)
                img = np.zeros((*self.dim, self.n_channels))
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.dim)
            
            # Augmentation
            if self.augment:
                img = self._augment_image(img)
            
            # Normalize to [0, 1]? No, MobileNet expects [-1, 1].
            # We let the model's Rescaling layer handle it.
            # Just return raw pixels (float32).
            X[i,] = img.astype('float32')

        return X

    def _augment_image(self, img):
        """Simple augmentation pipeline using OpenCV"""
        rows, cols, _ = img.shape
        
        # Random Flip
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            
        # Random Rotation
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img = cv2.warpAffine(img, M, (cols, rows))
        
        return img

def get_pokemon_dataset(batch_size=32, target_size=(224, 224), validation_split=0.2):
    """
    Utility to get Training and Validation generators.
    """
    # Load Metadata
    df = pd.read_csv(config.PROCESSED_DATA_DIR / "pokemon_metadata.csv")
    
    # Process Labels (Multi-Hot)
    from sklearn.preprocessing import MultiLabelBinarizer
    y_list = []
    for _, row in df.iterrows():
        types = [row['type1']]
        if pd.notna(row['type2']): types.append(row['type2'])
        y_list.append(types)
    
    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(y_list)
    
    # Image Paths
    image_paths = [str(config.RAW_DATA_DIR / f"{row['name']}.png") for _, row in df.iterrows()]
    
    # Split
    split_idx = int(len(image_paths) * (1 - validation_split))
    
    train_gen = PokemonDataGenerator(
        image_paths[:split_idx], 
        y_encoded[:split_idx], 
        batch_size=batch_size, 
        dim=target_size,
        augment=True
    )
    
    val_gen = PokemonDataGenerator(
        image_paths[split_idx:], 
        y_encoded[split_idx:], 
        batch_size=batch_size, 
        dim=target_size,
        augment=False
    )
    
    return train_gen, val_gen, mlb.classes_
