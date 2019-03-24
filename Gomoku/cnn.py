from constants import *
from formating import *
from torchsummary import summary

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # batch_size x 3 x 16 x 16
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        # batch_size x 16 x 16 x 16
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.norm1 = torch.nn.BatchNorm2d(16)
        # batch_size x 16 x 8 x 8
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # batch_size x 32 x 8 x 8
        # pool
        # batch_size x 32 x 4 x 4
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(64)
        self.conv5 = torch.nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.norm3 = torch.nn.BatchNorm2d(96)
        self.conv6 = torch.nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        # batch_size x 128 x 4 x 4
        self.norm4 = torch.nn.BatchNorm1d(128 * 4 * 4)
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 30 * 30)
        self.norm5 = torch.nn.BatchNorm1d(30 * 30)
        self.fc2 = torch.nn.Linear(30 * 30, 15 * 15)


    def forward(self, x):
        x = x.view(-1, 3, 16, 16)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.norm1(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.norm2(x)
        x = F.relu(self.conv5(x))
        x = self.norm3(x)
        x = F.relu(self.conv6(x))
        # x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.norm4(x)
        x = F.relu(self.fc1(x))
        x = self.norm5(x)
        x = self.fc2(x)
        return (x)

model = CNN()
summary(model, (3, 16, 16))

batch_s = 4096
train_s = 100
test_s = 10

l1_lmbd = 0.15
l2_lmbd = 0.25


def li_loss(layer, i):
    # print(layer.weight.data)
    return torch.norm(layer.weight.data, p=i)


def train(f, model, epoch):
    loss = 0
    data = []
    labels = []
    for i in range(train_s):
        while len(labels) < batch_s:
            log = f.readline()
            if len(log) == 0:
                continue
            d, l = log_to_data(log)
            data += d
            labels += l
        optimizer.zero_grad()
        batch = torch.stack(data[: batch_s])
        batch_lbl = torch.Tensor(labels[: batch_s])
        batch = batch.cuda().float()
        batch_lbl = batch_lbl.cuda().long()
        data = data[batch_s:]
        labels = labels[batch_s:]
        output = model(batch)
        loss = F.cross_entropy(output, batch_lbl) + l1_lmbd * li_loss(model.fc1, 1) \
               + l1_lmbd * li_loss(model.fc2, 1) + l2_lmbd * li_loss(model.fc1, 2) + l2_lmbd * li_loss(model.fc2, 2)
        loss.backward()
        optimizer.step()
        if i % (train_s // 5) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i, train_s,
                100. * i / train_s, loss.data.item()))


def test(f, model, err_log=None):
    with torch.no_grad():
        test_loss = 0
        test_size = 0
        test_correct = 0
        data = []
        labels = []
        for i in range(test_s):
            while len(labels) < batch_s:
                log = f.readline()
                d, l = log_to_data(log)
                data += d
                labels += l
            test_size += batch_s
            batch = torch.stack(data[: batch_s])
            batch_lbl = torch.Tensor(labels[: batch_s])
            batch, batch_lbl = Variable(batch), Variable(batch_lbl)
            batch, batch_lbl = batch.cuda().float(), batch_lbl.cuda().long()
            data = data[batch_s:]
            labels = labels[batch_s:]
            output = model(batch)
            test_loss += F.cross_entropy(output, batch_lbl, reduction='sum').data.item()
            pred = output.data.max(1, keepdim=True)[1]
            corr = pred.eq(batch_lbl.data.view_as(pred)).sum()
            test_correct += corr
        test_loss /= test_size
        test_correct_percent = 100. * test_correct / test_size
    print('\nTest set:  Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, test_correct, test_size, test_correct_percent))
    err_log['test'].append((test_loss, test_correct_percent))


def weights_init(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_normal_(model.weight, gain=1.4)
        model.bias.data.fill_(0.01)


def train_n_test(model, title, epoc_num, train=train):
    err_log = {'test': [], 'train': []}
    epochs = range(1, epoc_num)
    for epoch in epochs:
        with open('../input/good-games/good_ones.txt', "r") as f:
            train(f, model, epoch)
            print("epochs: ", epoch)
            test(f, model, err_log)
        f.close()




model = CNN()
model.load_state_dict(torch.load("upd2", map_location='cpu'))
model.eval()
d, l = process_log("white b4 c4 a5 c5 a3 c3")
print(d[0][0], d[0][1], d[0][2])
print()
out = model(d[0].float())
out = np.argmax(out.data)
print("out:", out // (n + 1) + 1, out % (n + 1) + 1)



