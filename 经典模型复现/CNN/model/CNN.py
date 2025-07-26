import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes,batch_size):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),      # batch_size*32*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # batch_size*32*14*14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),      # batch_size*64*14*14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                       # batch_size*64*7*7
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):   # x: batch_size*1*28*28
        x = self.feature_extractor(x)
        x = x.view(-1,64 * 7 * 7)  # Flatten the tensor batch_size*(64*7*7)
        x = self.classifier(x)
        return x