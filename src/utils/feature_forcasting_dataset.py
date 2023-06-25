from torch import Tensor
from numpy import concatenate
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view

class FeatureForcastingDataset(Dataset):
    def __init__(self, feature_series, target_feature, features:list, past_window:int=1, future_window:int=1):
        self.past_window = past_window
        self.future_window = future_window
        window_size = self.past_window + self.future_window 

        self.mean = feature_series.mean()
        self.std = feature_series.std()
        feature_series = (feature_series - self.mean)/self.std

        sliding_features = []
        for feature in features:
            sliding_window = sliding_window_view(feature_series[feature], window_shape=self.past_window)[:-self.future_window]
            sliding_features.append(sliding_window)
        self.features = Tensor(concatenate(sliding_features, axis=1))
        self.targets = Tensor(sliding_window_view(feature_series[target_feature], window_shape=window_size)[:,-self.future_window:])
        del sliding_window

        # dataset = sliding_window_view(feature_series[target_feature], window_shape=window_size)
        # dataset = Tensor(dataset)
        # self.features = dataset[:,:-self.future_window]
        # self.targets = dataset[:,-self.future_window:]

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return self.features[index], self.targets[index]