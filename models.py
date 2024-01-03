import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNetwork(nn.Module):
  def __init__(self, input_shape, action_space, first_layer_size=512, second_layer_size=256):
    super().__init__()
    self.first_layer = nn.Linear(input_shape[0], first_layer_size)
    self.second_layer = nn.Linear(first_layer_size, second_layer_size)
    self.value_head = nn.Linear(second_layer_size, 1)
    self.policy_head = nn.Linear(second_layer_size, action_space)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(self.device)

  def __call__(self, observations):
    x = F.relu(self.first_layer(observations))
    x = F.relu(self.second_layer(x))
    value = F.tanh(self.value_head(x))
    log_policy = F.log_softmax(self.policy_head(x), dim=-1)
    return value, log_policy

  def value_forward(self, observation):
    self.eval()
    with torch.no_grad():
      x = F.relu(self.first_layer(observation))
      x = F.relu(self.second_layer(x))
      value = F.tanh(self.value_head(x))
      return value

  def policy_forward(self, observation):
    self.eval()
    with torch.no_grad():
      x = F.relu(self.first_layer(observation))
      x = F.relu(self.second_layer(x))
      log_policy = F.softmax(self.policy_head(x), dim=-1)
      return log_policy


class ConvolutionalNetwork(nn.Module):
  def __init__(self, input_shape, action_space, first_linear_size=512, second_linear_size=256):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=1)
    self.conv2 = nn.Conv2d(32, 32, kernel_size=1)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=1)
    self.dropout = nn.Dropout2d(p=0.3)
    self.fc1 = nn.Linear(3 * 3 * 64, first_linear_size)
    self.fc2 = nn.Linear(first_linear_size, second_linear_size)
    self.value_head = nn.Linear(second_linear_size, 1)
    self.policy_head = nn.Linear(second_linear_size, action_space)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(self.device)

  def __call__(self, observations):
    x = F.relu(self.conv1(observations))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = self.dropout(x)
    x = x.view(-1, 3 * 3 * 64)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    value = F.tanh(self.value_head(x))
    log_policy = F.log_softmax(self.policy_head(x), dim=-1)
    return value, log_policy

  def value_forward(self, observation):
    with torch.no_grad():
      x = F.relu(self.conv1(observation))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = x.view(-1, 3 * 3 * 64)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      value = F.tanh(self.value_head(x))
      return value[0]

  def policy_forward(self, observation):
    with torch.no_grad():
      x = F.relu(self.conv1(observation))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = x.view(-1, 3 * 3 * 64)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      log_policy = F.softmax(self.policy_head(x), dim=-1)
      return log_policy[0]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class Connect4Network(nn.Module):
    def __init__(self, input_shape, action_space):
        super().__init__()
        num_filters = 64
        self.conv1 = nn.Conv2d(input_shape[0], num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.resblock1 = ResidualBlock(num_filters)
        self.resblock2 = ResidualBlock(num_filters)

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * input_shape[1] * input_shape[2], action_space)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * input_shape[1] * input_shape[2], 64)
        self.value_fc2 = nn.Linear(64, 1)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.to(self.device)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        # Apply residual blocks
        x = self.resblock1(x)
        x = self.resblock2(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        log_policy = F.log_softmax(self.policy_fc(policy), dim=-1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return value.cpu(), log_policy.cpu()

    def policy_forward(self, x):
        self.eval()
        x = x.unsqueeze(0) if x.dim() == 3 else x  # Add batch dimension if missing
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.resblock1(x)
            x = self.resblock2(x)
            policy = F.relu(self.policy_bn(self.policy_conv(x)))
            policy = policy.view(policy.size(0), -1)
            log_policy = F.softmax(self.policy_fc(policy), dim=-1)
            return log_policy.squeeze(0).cpu()  # Remove batch dimension for single instance

    def value_forward(self, x):
        self.eval()
        x = x.unsqueeze(0) if x.dim() == 3 else x  # Add batch dimension if missing
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.resblock1(x)
            x = self.resblock2(x)
            value = F.relu(self.value_bn(self.value_conv(x)))
            value = value.view(value.size(0), -1)
            value = F.relu(self.value_fc1(value))
            value = torch.tanh(self.value_fc2(value))
            return value.squeeze(0).cpu()
