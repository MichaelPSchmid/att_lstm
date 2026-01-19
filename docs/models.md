# Modellarchitekturen

## Übersicht

Alle Modelle sind als PyTorch Lightning Module implementiert und folgen einem einheitlichen Interface.

| Modell | Datei | Attention |
|--------|-------|-----------|
| LSTMModel | `model/LSTM.py` | Keine |
| LSTMAttentionModel (Simple) | `model/LSTM_attention.py` | Linear → Softmax |
| LSTMAttentionModel (Additive) | `model/diff_attention/additive_attention.py` | Bahdanau |
| LSTMScaleDotAttentionModel | `model/diff_attention/scaled_dot_product.py` | Scaled Dot-Product |
| CNNModel | `model/CNN_eval.py` | Keine |

---

## 1. LSTMModel (Baseline)

**Datei:** `model/LSTM.py`

### Architektur

```
Input (batch, 50, 5)
        ↓
    LSTM Layer(s)
        ↓
  Last Hidden State (batch, hidden_size)
        ↓
  Fully Connected (batch, 1)
        ↓
    Output
```

### Implementierung

```python
class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lr=0.001):
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
```

### Forward Pass

```python
def forward(self, x):
    # Initialisiere Hidden States mit Nullen
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

    # LSTM Forward
    lstm_output, (hn, _) = self.lstm(x, (h0, c0))

    # Nutze letzten Hidden State für Vorhersage
    output = self.fc(hn[-1])
    return output
```

### Eigenschaften
- Nutzt nur den **letzten Hidden State** (`hn[-1]`) für die Vorhersage
- Keine explizite Gewichtung der Zeitschritte
- Einfachste Architektur als Baseline

---

## 2. LSTMAttentionModel (Simple Attention)

**Datei:** `model/LSTM_attention.py`

### Architektur

```
Input (batch, 50, 5)
        ↓
    LSTM Layer(s)
        ↓
  All Hidden States (batch, 50, hidden_size)
        ↓
  Attention: Linear(hidden_size → 1) + Softmax
        ↓
  Weighted Sum → Context Vector (batch, hidden_size)
        ↓
  Fully Connected (batch, 1)
        ↓
    Output
```

### Attention-Mechanismus

```python
# Attention Layer
self.attention = nn.Linear(hidden_size, 1)

# Forward
attention_scores = self.attention(lstm_output)      # (batch, seq_len, 1)
attention_weights = torch.softmax(attention_scores, dim=1)
context_vector = torch.sum(attention_weights * lstm_output, dim=1)
```

### Eigenschaften
- Lernt **einen Gewichtungsvektor** über alle Zeitschritte
- Einfache lineare Transformation → Softmax
- Context Vector ist gewichtete Summe aller LSTM-Outputs

---

## 3. LSTMAttentionModel (Additive/Bahdanau Attention)

**Datei:** `model/diff_attention/additive_attention.py`

### Architektur

```
Input (batch, 50, 5)
        ↓
    LSTM Layer(s)
        ↓
  All Hidden States (batch, 50, hidden_size)
        ↓
  Additive Attention (Bahdanau)
        ↓
  Context Vector (batch, hidden_size)
        ↓
  Fully Connected (batch, 1)
        ↓
    Output
```

### Attention-Mechanismus

Bahdanau Attention berechnet Scores zwischen allen Paaren von Zeitschritten:

```python
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size, attention_size=128):
        self.w = nn.Linear(hidden_size, attention_size, bias=False)  # W
        self.u = nn.Linear(hidden_size, attention_size, bias=False)  # U
        self.v = nn.Parameter(torch.empty(attention_size, 1))        # v

    def forward(self, lstm_output):
        # e_ij = v^T · tanh(W·h_i + U·h_j)
        w_x = self.w(lstm_output)  # Transform alle Zeitschritte
        u_x = self.u(lstm_output)  # Transform alle Zeitschritte

        e = torch.tanh(w_x + u_x)  # Additive Kombination
        e = torch.matmul(e, self.v)  # Score via v

        attention_weights = F.softmax(e, dim=-1)
        context_vector = torch.bmm(attention_weights, lstm_output)
        return context_vector, attention_weights
```

### Formel

$$e_{ij} = v^T \cdot \tanh(W \cdot h_i + U \cdot h_j)$$

$$\alpha_{ij} = \text{softmax}(e_{ij})$$

### Eigenschaften
- Berechnet **paarweise Attention** zwischen allen Zeitschritten
- Drei lernbare Parameter: W, U, v
- Komplexer als Simple Attention, aber expressiver
- Output: (batch, seq_len, seq_len) Attention Matrix

---

## 4. LSTMScaleDotAttentionModel (Scaled Dot-Product)

**Datei:** `model/diff_attention/scaled_dot_product.py`

### Architektur

```
Input (batch, 50, 5)
        ↓
    LSTM Layer(s)
        ↓
  All Hidden States (batch, 50, hidden_size)
        ↓
  Scaled Dot-Product Attention
        ↓
  Context Vector (batch, hidden_size)
        ↓
  Fully Connected (batch, 1)
        ↓
    Output
```

### Attention-Mechanismus

```python
class ScaledDotProductAttention(nn.Module):
    def forward(self, x):
        # Self-Attention: Q = K = V = x
        e = torch.bmm(x, x.permute(0, 2, 1))  # (b, s, s)
        e = e / math.sqrt(self.hidden_size)   # Skalierung

        attention = F.softmax(e, dim=-1)

        # Global Attention: Mittelwert über alle Zeitschritte
        global_attention = torch.mean(attention, dim=1)
        context_vector = torch.bmm(global_attention.unsqueeze(1), x)

        return context_vector, global_attention
```

### Formel

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Hier: Q = K = V = LSTM Output (Self-Attention)

### Eigenschaften
- **Self-Attention** ohne separate Q, K, V Projektionen
- Skalierung mit √d_k verhindert zu große Dot-Products
- Globale Attention durch Mittelung über Zeitdimension
- Keine zusätzlichen lernbaren Parameter im Attention-Modul

---

## 5. CNNModel

**Datei:** `model/CNN_eval.py`

### Architektur

```
Input (batch, 50, 5)
        ↓
  Permute → (batch, 5, 50)
        ↓
  Conv1D + BatchNorm + ReLU
        ↓
  Conv1D + BatchNorm + ReLU
        ↓
  Global Average Pooling
        ↓
  Fully Connected (batch, 1)
        ↓
    Output
```

### Implementierung

```python
class CNNModel(pl.LightningModule):
    def __init__(self, input_size, num_filters, kernel_size, output_size, lr=0.001):
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding="same")
        self.batchnorm1 = nn.BatchNorm1d(num_filters)
        self.batchnorm2 = nn.BatchNorm1d(num_filters)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.global_avg_pool(x).squeeze(2)
        return self.fc(x)
```

### Eigenschaften
- **1D Convolution** über die Zeitdimension
- `padding="same"` erhält Sequenzlänge
- **Global Average Pooling** statt Flatten (parametereffizient)
- Batch Normalization für stabiles Training
- Keine rekurrente Struktur → potenziell schnelleres Training

---

## Modellvergleich

| Eigenschaft | LSTM | LSTM+Simple | LSTM+Additive | LSTM+ScaledDot | CNN |
|-------------|------|-------------|---------------|----------------|-----|
| Rekurrent | ✓ | ✓ | ✓ | ✓ | ✗ |
| Attention | ✗ | ✓ | ✓ | ✓ | ✗ |
| Attention-Parameter | 0 | hidden_size | 3×attention_size | 0 | 0 |
| Paarweise Attention | ✗ | ✗ | ✓ | ✓ | ✗ |
| Interpretierbarkeit | Niedrig | Mittel | Hoch | Hoch | Niedrig |

---

## Gemeinsame Komponenten

### Loss Function
Alle Modelle verwenden **MSELoss** (Mean Squared Error):
```python
self.criterion = nn.MSELoss()
```

### Optimizer
Standard: **Adam** mit konfigurierbarer Learning Rate:
```python
optim.Adam(self.parameters(), lr=self.lr)
```

### Learning Rate Scheduler (nur LSTMModel)
```python
optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5
)
```

---

## Hyperparameter

| Parameter | Beschreibung | Typische Werte |
|-----------|--------------|----------------|
| `input_size` | Anzahl Features | 5 |
| `hidden_size` | LSTM Hidden Dimension | 32-128 |
| `num_layers` | Anzahl LSTM-Schichten | 1-5 |
| `output_size` | Vorhersage-Dimension | 1 |
| `lr` | Learning Rate | 1e-5 bis 1e-3 |
| `num_filters` | CNN Filter (nur CNNModel) | 32-128 |
| `kernel_size` | CNN Kernel (nur CNNModel) | 3-7 |
