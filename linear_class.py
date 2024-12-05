import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
class DeepLearn(nn.Module):
    def __init__(self,in_features=512, out_features=100):
        super().__init__()
        self.linear1 = nn.Linear(in_features,out_features)
    def forward(self,x):
        x = self.linear1(x)
        return x

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,train_path="D:\\code\\train\\train.csv"):
        ds_train=pd.read_csv(train_path)
        self.data=ds_train.iloc[:,:-1]
        self.labels=ds_train.iloc[:,-1]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx,:]), torch.tensor(self.labels.iloc[idx])

train_dataloader=torch.utils.data.DataLoader(ImageDataset(),batch_size=256,shuffle=True)
model = DeepLearn()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.to(torch.float32)
print(torch.cuda.is_available())
for epoch in range(10):
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    # 根据训练数据计算正确率
    correct = 0
    total = 0
    model.eval()
    for inputs, labels in train_dataloader:
        inputs = inputs.to(torch.float32)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        predicted= torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}%")
    model.train()

