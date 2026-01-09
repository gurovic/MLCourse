# Мультимодальное обучение

## 🟢 Основы (Basic Level)

### Введение в Multimodal Learning

**Multimodal Learning** - обучение моделей, которые работают с несколькими типами данных одновременно (текст, изображения, аудио, видео).

**Примеры задач:**
- Image captioning (изображение → текст)
- Visual Question Answering (изображение + вопрос → ответ)
- Text-to-Image generation (текст → изображение)
- Video understanding (видео + аудио → описание)

### Простое объединение модальностей

Наивный подход: отдельные энкодеры + конкатенация.

```python
import torch
import torch.nn as nn
from torchvision import models

class SimpleMultimodalModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleMultimodalModel, self).__init__()
        
        # Image encoder (ResNet без последнего слоя)
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        image_feature_dim = 512
        
        # Text encoder (простой LSTM)
        self.text_encoder = nn.LSTM(
            input_size=300,  # Размер word embeddings
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        text_feature_dim = 256
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(image_feature_dim + text_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, image, text):
        # Image features
        img_features = self.image_encoder(image)
        img_features = img_features.view(img_features.size(0), -1)
        
        # Text features
        _, (text_features, _) = self.text_encoder(text)
        text_features = text_features[-1]  # Последний hidden state
        
        # Concatenate
        combined = torch.cat([img_features, text_features], dim=1)
        
        # Classification
        output = self.fusion(combined)
        return output

# Использование
model = SimpleMultimodalModel(num_classes=10)

# Пример входных данных
batch_size = 4
image = torch.randn(batch_size, 3, 224, 224)  # Batch изображений
text = torch.randn(batch_size, 20, 300)  # Batch текстов (20 слов, embedding 300)

output = model(image, text)
print(f"Output shape: {output.shape}")  # [4, 10]
```

### Image Captioning (базовый)

Классическая задача: CNN encoder + RNN decoder.

```python
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(ImageCaptioningModel, self).__init__()
        
        # Image encoder (CNN)
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Linear layer to project image features to embedding space
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        
        # Text decoder (LSTM)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, images, captions, lengths):
        # Encode images
        features = self.encoder(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        
        # Embed captions
        embeddings = self.embed(captions)
        
        # Concatenate image features and caption embeddings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        # Pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        
        # LSTM forward
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        
        return outputs
    
    def sample(self, features, max_length=20):
        """Generate captions for given image features."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None
        
        for i in range(max_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

# Обучение
model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=10000)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Forward pass
# outputs = model(images, captions, lengths)
# loss = criterion(outputs, targets)
# 
# # Backward pass
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
```

## 🟡 Средний уровень (Intermediate Level)

### CLIP (Contrastive Language-Image Pre-training)

Современный подход к мультимодальному обучению от OpenAI.

```python
import clip
from PIL import Image

# Загрузка предобученной модели CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Загрузка изображения
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)

# Текстовые описания
text_descriptions = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a car"
]

text = clip.tokenize(text_descriptions).to(device)

# Получение embeddings
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # Нормализация
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Cosine similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Результаты
for desc, prob in zip(text_descriptions, similarity[0]):
    print(f"{desc}: {prob.item():.2%}")
```

### Custom CLIP-like Model

Упрощенная реализация CLIP-подобной модели.

```python
class CLIPModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=512, temperature=0.07):
        super(CLIPModel, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
        # Projection heads
        self.image_projection = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_projection = nn.Linear(text_encoder.output_dim, embed_dim)
        
    def forward(self, images, texts):
        # Encode
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        
        # Project
        image_embed = self.image_projection(image_features)
        text_embed = self.text_projection(text_features)
        
        # Normalize
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        
        return image_embed, text_embed
    
    def contrastive_loss(self, image_embed, text_embed):
        # Cosine similarity matrix
        logits = (image_embed @ text_embed.T) / self.temperature
        
        # Labels (diagonal should be high)
        batch_size = image_embed.shape[0]
        labels = torch.arange(batch_size, device=image_embed.device)
        
        # Symmetric loss
        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        return loss

# Обучение
# for images, texts in dataloader:
#     image_embed, text_embed = model(images, texts)
#     loss = model.contrastive_loss(image_embed, text_embed)
#     
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
```

### Visual Question Answering (VQA)

```python
class VQAModel(nn.Module):
    def __init__(self, num_answers, embed_dim=512):
        super(VQAModel, self).__init__()
        
        # Image encoder
        resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_projection = nn.Linear(2048, embed_dim)
        
        # Question encoder (BERT)
        from transformers import BertModel
        self.question_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.question_projection = nn.Linear(768, embed_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        
        # Answer classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_answers)
        )
        
    def forward(self, images, questions_input_ids, questions_attention_mask):
        # Encode image
        img_features = self.image_encoder(images)
        img_features = img_features.view(img_features.size(0), -1)
        img_embed = self.image_projection(img_features)
        
        # Encode question
        question_output = self.question_encoder(
            input_ids=questions_input_ids,
            attention_mask=questions_attention_mask
        )
        question_embed = question_output.last_hidden_state[:, 0, :]  # [CLS] token
        question_embed = self.question_projection(question_embed)
        
        # Cross-attention: question attends to image
        img_embed = img_embed.unsqueeze(0)  # (1, batch, embed_dim)
        question_embed = question_embed.unsqueeze(0)
        
        attended_features, _ = self.attention(
            query=question_embed,
            key=img_embed,
            value=img_embed
        )
        
        attended_features = attended_features.squeeze(0)
        
        # Classify answer
        answer_logits = self.classifier(attended_features)
        return answer_logits
```

## 🔴 Продвинутый уровень (Expert Level)

### Multimodal Transformers

Unified Transformer для нескольких модальностей.

```python
class MultimodalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6):
        super(MultimodalTransformer, self).__init__()
        
        # Modality embeddings
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.image_patch_embed = nn.Linear(3*16*16, embed_dim)  # 16x16 patches
        
        # Modality type embeddings
        self.text_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.image_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Position embeddings
        self.position_embed = nn.Parameter(torch.randn(1, 512, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.mlm_head = nn.Linear(embed_dim, vocab_size)  # Masked Language Modeling
        self.itm_head = nn.Linear(embed_dim, 2)  # Image-Text Matching
        
    def forward(self, text_tokens, image_patches, mask=None):
        batch_size = text_tokens.size(0)
        
        # Text embeddings
        text_embed = self.text_embed(text_tokens)
        text_embed = text_embed + self.text_type_embed
        
        # Image embeddings
        image_embed = self.image_patch_embed(image_patches)
        image_embed = image_embed + self.image_type_embed
        
        # Concatenate modalities
        combined = torch.cat([text_embed, image_embed], dim=1)
        
        # Add position embeddings
        seq_length = combined.size(1)
        combined = combined + self.position_embed[:, :seq_length, :]
        
        # Transformer
        combined = combined.transpose(0, 1)  # (seq, batch, embed)
        output = self.transformer(combined, src_key_padding_mask=mask)
        output = output.transpose(0, 1)  # (batch, seq, embed)
        
        # Task-specific heads
        mlm_logits = self.mlm_head(output)
        itm_logits = self.itm_head(output[:, 0, :])  # [CLS] token
        
        return mlm_logits, itm_logits
```

### Flamingo-style Architecture

Few-shot learning с frozen LM и vision encoder.

```python
class FlamingoBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(FlamingoBlock, self).__init__()
        
        # Cross-attention to visual features
        self.cross_attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        
        # Self-attention (frozen LM layer)
        self.self_attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm3 = nn.LayerNorm(dim)
        
        # Gating (trainable)
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, text_features, visual_features):
        # Cross-attention with gating
        cross_out, _ = self.cross_attn(
            query=text_features,
            key=visual_features,
            value=visual_features
        )
        text_features = text_features + torch.tanh(self.alpha) * self.norm1(cross_out)
        
        # Self-attention (from frozen LM)
        self_out, _ = self.self_attn(text_features, text_features, text_features)
        text_features = text_features + self.norm2(self_out)
        
        # FFN
        text_features = text_features + self.norm3(self.ffn(text_features))
        
        return text_features
```

### Audio-Visual Multimodal Learning

```python
class AudioVisualModel(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioVisualModel, self).__init__()
        
        # Visual encoder (3D CNN for video)
        self.visual_encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Audio encoder (1D CNN for spectrogram)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fusion with attention
        self.fusion_attention = nn.MultiheadAttention(256, num_heads=4)
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, video, audio):
        # Encode modalities
        visual_feat = self.visual_encoder(video).view(video.size(0), -1)
        audio_feat = self.audio_encoder(audio).view(audio.size(0), -1)
        
        # Concatenate and project
        combined = torch.cat([visual_feat, audio_feat], dim=1)
        combined = combined.unsqueeze(0)  # (1, batch, dim)
        
        # Self-attention for fusion
        fused, _ = self.fusion_attention(combined, combined, combined)
        fused = fused.squeeze(0)
        
        # Classification
        output = self.classifier(fused)
        return output
```

## Ссылки

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Flamingo Paper](https://arxiv.org/abs/2204.14198)
- [ViLBERT Paper](https://arxiv.org/abs/1908.02265)
- [BLIP Paper](https://arxiv.org/abs/2201.12086)
- [Hugging Face Multimodal](https://huggingface.co/docs/transformers/model_doc/vision-text-dual-encoder)

## Datasets

- [COCO Captions](https://cocodataset.org/#captions-2015)
- [VQA v2](https://visualqa.org/)
- [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/)
- [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/)
