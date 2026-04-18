import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, average_precision_score
from torch_geometric.nn import GCNConv, GATConv, JumpingKnowledge
from torch_geometric.data import DataLoader


# 定义GNN模型
class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1, dropout=0.5):
        super(GATModel, self).__init__()

        # 初始化图注意力层 (GATConv)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))

        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))

        # 跳跃连接 (Jumping Knowledge, JK)
        self.jk = JumpingKnowledge(mode='cat')

        # 全连接层，最后输出分类结果
        self.fc = torch.nn.Linear(num_layers * hidden_channels * heads, out_channels)

        # Dropout for regularization
        self.dropout = dropout

        # 残差连接层：用于调整维度
        self.residual_connections = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:  # 第一层的输入与隐藏层的维度不一定相同
                self.residual_connections.append(torch.nn.Linear(in_channels, hidden_channels * heads))
            else:
                self.residual_connections.append(torch.nn.Identity())  # 如果维度匹配，直接使用Identity

    def forward(self, x, edge_index, edge_attr=None):
        # 存储每层的节点嵌入
        layer_outs = []

        # 逐层图注意力卷积
        for i, conv in enumerate(self.convs):
            identity = x  # 保存残差连接的输入

            # 图卷积操作
            x = F.elu(conv(x, edge_index, edge_attr))

            # 残差连接：调整输入维度或直接相加
            if isinstance(self.residual_connections[i], torch.nn.Linear):
                identity = self.residual_connections[i](identity)
            x = x + identity  # 残差连接

            # 存储每一层的输出
            layer_outs.append(x)

            # Dropout 正则化
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 使用跳跃连接 (Jumping Knowledge)
        x = self.jk(layer_outs)

        return self.fc(x)

# 训练函数
# 训练函数
def train(model, optimizer, train_data, device):
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x.to(device), train_data.edge_index.to(device), train_data.edge_attr.to(device))
    loss = F.cross_entropy(out, train_data.y.to(device))  # 交叉熵损失
    loss.backward()
    optimizer.step()
    return loss.item()


# 测试函数，计算损失值、准确率、精准度、AUC、AUPRC
def test(model, test_data, device):
    model.eval()
    out = model(test_data.x.to(device), test_data.edge_index.to(device), test_data.edge_attr.to(device))

    # 计算损失
    loss = F.cross_entropy(out, test_data.y.to(device)).item()

    # 获取预测结果和真实标签
    y_true = test_data.y.cpu().numpy()
    y_pred = out.argmax(dim=1).cpu().numpy()
    y_prob = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()  # 计算类别1的概率

    # 计算分类准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 计算精准度
    precision = precision_score(y_true, y_pred)

    # 计算AUC（ROC AUC）
    auc = roc_auc_score(y_true, y_prob)

    # 计算AUPRC
    auprc = average_precision_score(y_true, y_prob)

    return loss, accuracy, precision, auc, auprc


# 加载图数据
train_graph = torch.load('./data/train/train_graph_k0.pt')
test_graph = torch.load('./data/train/val_graph_k0.pt')
# 将标签为-1的映射为0
train_graph.y = torch.where(train_graph.y == -1, torch.tensor(0), train_graph.y)
test_graph.y = torch.where(test_graph.y == -1, torch.tensor(0), test_graph.y)

# 超参数设置
in_channels = train_graph.num_node_features  # 输入特征维度
print(in_channels)
hidden_channels = 32  # 隐藏层特征维度
out_channels = 2  # 输出类别数
num_layers = 4  # GCN层数
dropout = 0.5  # Dropout比例

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATModel(train_graph.num_node_features, hidden_channels, 2,
                                     num_layers, heads=4).to(device)
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
best_test_loss = float('inf')
best_model_state = None
for epoch in range(1, 5001):  # 每组参数最多训练 500 个 epoch
    train_loss = train(model, optimizer, train_graph, device)
    test_loss, accuracy, precision,auc, auprc = test(model, test_graph, device)
    print(
        f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f},Precision: {precision:.4f}, AUC: {auc:.4f}, AUPRC: {auprc:.4f}')

    # 检查是否有改进
    if min(best_test_loss, test_loss * 1.05) >= test_loss:
        best_test_loss = test_loss
        epochs_no_improve = 0  # 重置没有改进的epoch数
        best_model_state = model.state_dict()  # 保存当前最优模型
        torch.save(best_model_state, './best_model_k0.pth')
    else:
        epochs_no_improve += 1

    # 如果没有改进的epoch数超过patience，停止训练
    if epochs_no_improve >= 10:
        print(f"早停于Epoch {epoch}，没有改进超过 {10} 个epoch.")
        break
