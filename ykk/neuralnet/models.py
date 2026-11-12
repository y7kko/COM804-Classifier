import torch
import torch.nn as nn

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )
#         self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits


class SalamonCNN(nn.Module):
    def __init__(self,dropout=.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.LazyConv2d(24, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),


            nn.LazyConv2d(48, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),


            nn.LazyConv2d(48, kernel_size=5),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),      
            nn.Dropout(dropout),
            nn.LazyLinear(64), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LazyLinear(10), 
        )


    def forward(self, x):
        return self.classifier(self.features(x))



class LeNet5(nn.Module):
    def __init__(self,dropout=.5):
        """Inspirado em, fiz alterações

        Args:
            dropout (float, optional): _description_. Defaults to .5.
        """
        super().__init__()
        # Nx-|K|+1 = 20 - 5
        self.sequence = nn.Sequential(
            # C1
            nn.Conv2d(1, 6, kernel_size=28),   # (B, 6, 33, 103)
            nn.ReLU(inplace=True),

            # S2
            nn.MaxPool2d(kernel_size=2),      # !(B, 6, 16, 51)
            
            # C3
            nn.Conv2d(6, 16, kernel_size=10), # (B, 16, 7, 42)
            nn.ReLU(inplace=True),

            # S4
            nn.MaxPool2d(2),                  # !(B, 16, 3, 21)
            nn.Flatten(),                      # (B, 1008)

            # F5
            nn.Linear(16 * 3 * 21, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            # F6
            nn.Linear(84, 10),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )


    def forward(self, x):
        x = self.sequence(x)
        return x
