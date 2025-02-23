import os
import random
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from tqdm import tqdm 
from torch.optim import Adam


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.scale = dim ** -0.5
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return rearrange(out, 'b h n d -> b n (h d)')
    


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class Vit(nn.Module):
    def __init__(self, image_size=(50, 80), patch_size=10, dim=512, depth=16, heads=8, mlp_dim=1024, dropout=0.1, emb_dropout=0.1, classes=10):
        super().__init__()

        patch_dim = patch_size * patch_size
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)            
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, classes)
        )


    def forward(self, x):
        x = self.patch_embedding(x)
        b, n, _ = x.shape
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        return self.mlp_head(x[:, 0])


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 获取所有图片路径和标签（文件名）
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg')):  # 支持更多图像格式
                self.image_paths.append(os.path.join(image_dir, filename))
                self.labels.append(filename.split('.')[0])  # 假设文件名即为标签

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.tensor(int(self.labels[idx]))

        try:
            image = Image.open(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            image = self.transform(image)
        except Exception as e:
            print(f"警告：处理图像 {image_path} 时出错：{e}，返回空样本。")
            return torch.zeros(1), ""
        return image, label



def create_data_loaders(image_dir, size,  train_ratio=0.7, val_ratio=0.15, batch_size=32, num_workers=4):
    # 定义图像变换（数据预处理和增强）
    # 示例：调整大小、转换为 Tensor
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # 先不标准化
    ])
    
    # 创建数据集实例
    dataset = ImageDataset(image_dir, transform=transform)
    
    # 计算划分数量
    test_ratio = 1 - train_ratio - val_ratio
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':


    learning_rate = 1e-4
    epochs = 50
    batch_size = 64
    image_size = (50, 80)
    patch_size = 10
    dim = 512
    depth = 16
    mlp_dim = 1024
    dropout = 0.1
    emb_dropout = 0.1
    classes = 10
    heads = 2
    image_directory = 'zfw'  # 替换为您的图片文件夹
    data_loaders = create_data_loaders(image_directory, image_size)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for batch_idx, (images, labels) in enumerate(data_loaders['train']):
        print(f"批次 {batch_idx+1}:")
        print("  图像形状:", images.shape)  # 输出应该是 [batch_size, 784] (如果图像大小是 28x28)
        print("  标签:", labels)
        break  # 只查看第一个批次
    # 模型实例化
    model = Vit(image_size, patch_size, dim, depth, heads, mlp_dim, dropout, emb_dropout, classes).to(device)

    # 优化器和损失函数
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # 学习率调度器（可选）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 2. 训练循环
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        
        # 使用 tqdm 显示进度条
        with tqdm(total=len(data_loaders['train']), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for i, (images, labels) in enumerate(data_loaders['train']):
                
                images = images.to(device)
                labels = labels.to(device)
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (i + 1)})
                pbar.update(1)
        
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(data_loaders["train"])}')

        # 验证 (可选)
        model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        with torch.no_grad():  # 在验证阶段，我们不需要计算梯度
            for images, labels in data_loaders['val']:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy on validation set: {100 * correct / total}%')

    # 测试 (可选)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loaders['test']:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total}%')
    print('Finished Training')