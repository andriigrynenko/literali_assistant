import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

characters = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я', 'є', 'і', 'ї', 'ґ']

class CharNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(characters))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def img_to_tensor(img):
        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize((torch.mean(img_tensor)), (torch.std(img_tensor)))(img_tensor)
        return img_tensor.view((1, 1, 32, 32))

    @staticmethod
    def create_from_file(path):
        net = CharNet()
        net.load_state_dict(torch.load('char_net0.pt'))
        return net

    def guess_character(self, img):
        with torch.no_grad(): 
            tensor = CharNet.img_to_tensor(img)
            output = self(tensor)
            if output[0][torch.max(output,1)[1]] < 15:
                return None
            return characters[torch.max(output,1)[1]]
