import torch
import torch.nn as nn
import torchvision.models as models

class PatientClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(PatientClassifier, self).__init__()

        # Load DenseNet201 pretrained on ImageNet
        densenet = models.densenet201(pretrained=pretrained)
        self.feature_extractor = densenet.features  # outputs shape (1920, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))    # shape (1920, 1, 1)
        self.flatten = nn.Flatten()                 # shape (1920,)

        # Final classifier head after mean pooling over all images
        self.classifier = nn.Sequential(
            nn.Linear(1920, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Output: probability of 'improved'
        )

    def forward(self, x):
        # x shape: (N, 1, 224, 224) → grayscale images for a patient
        N = x.shape[0]

        # Duplicate channels to convert grayscale to 3-channel for DenseNet
        x = x.repeat(1, 3, 1, 1)  # shape: (N, 3, 224, 224)

        # Extract features from each image
        features = []
        for i in range(N):
            f = self.feature_extractor(x[i].unsqueeze(0))
            f = self.pool(f)
            f = self.flatten(f)
            features.append(f)

        features = torch.stack(features)            # shape: (N, 1920)
        patient_repr = torch.mean(features, dim=0)  # mean over N images → (1920,)

        output = self.classifier(patient_repr)      # shape: (1,)
        return output.squeeze()  # return scalar

# Example usage:
# model = PatientClassifier()
# output = model(image_tensor)  # image_tensor shape: (N, 1, 224, 224)
