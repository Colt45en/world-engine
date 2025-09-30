import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchcrf import CRF  # pip install pytorch-crf for CRF support
    HAS_CRF = True
except Exception:
    HAS_CRF = False

def sinusoidal_positions(n_pos: int, d: int, device=None):
    """[1, n_pos, d] sinusoidal PE, standard transformer-style."""
    pe = torch.zeros(n_pos, d, device=device)
    position = torch.arange(0, n_pos, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, n_pos, d]

class Residual(nn.Module):
    def __init__(self, mod, d_model, p=0.1):
        super().__init__()
        self.mod = mod
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p)
    def forward(self, x, *args, **kwargs):
        return self.norm(x + self.drop(self.mod(x, *args, **kwargs)))

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hidden, d_out)
        )
    def forward(self, x): return self.net(x)

class WorldEngine(nn.Module):
    """
    WE-v1: sentence-level 'roots' z plus token-level roles.
    Inputs
      tok_ids:  Long[B,N]       (token ids; 0 = pad)
      pos_ids:  Long[B,N]       (pos ids; 0 = pad)
      feat_rows:Float[B,N,K]    (interpretable features per token; can be zeros for pads)
      lengths:  Long[B]         (true lengths for masking)
      edge_index: Long[2, E]    (global COO over batch with sentence offsets)  (optional)
      edge_type:  Long[E]       (relation ids aligned with edge_index)        (optional)
    """
    def __init__(self, vocab_size, d_model, k_feats, n_pos, n_rels,
                 d_tok=None, d_pos=None, d_feat=None, p_drop=0.1,
                 use_transformer=True, n_layers=2, n_heads=4,
                 use_gnn=False, use_crf=False, num_role_labels=3):
        super().__init__()
        d_tok  = d_tok  or d_model // 2
        d_posE = d_pos  or d_model // 4
        d_feat = d_feat or d_model - d_tok - d_posE
        assert d_tok + d_posE + d_feat == d_model, "dims must sum to d_model"

        self.emb_tok = nn.Embedding(vocab_size, d_tok, padding_idx=0)
        self.emb_pos = nn.Embedding(n_pos,     d_posE, padding_idx=0)
        self.W_feat  = nn.Linear(k_feats, d_feat, bias=False)

        # positional encoding buffer (max length set at init; can be extended)
        self.register_buffer("pe", sinusoidal_positions(512, d_model), persistent=False)

        enc_layers = []
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                dropout=p_drop, batch_first=True, activation="gelu", norm_first=True
            )
            self.enc_seq = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            self.enc_seq = nn.Identity()

        # optional GNN hook: keep interface even if disabled
        self.use_gnn = use_gnn
        if use_gnn:
            # simple message-passing via attention over edges (no external deps)
            self.edge_rel_emb = nn.Embedding(n_rels, d_model)
            self.msg_proj = nn.Linear(2*d_model, d_model)
            self.msg_norm = nn.LayerNorm(d_model)

        # latent head (sentence-level roots)
        self.enc_lat = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 32)  # z
        )

        # reconstruct interpretable features (either sentence-avg or token-avg target)
        self.dec_feat = nn.Linear(32, k_feats)

        # token role tagging head
        self.role_head = nn.Linear(d_model, num_role_labels)
        self.use_crf = use_crf and HAS_CRF and (num_role_labels > 1)
        if self.use_crf:
            self.crf = CRF(num_role_labels, batch_first=True)

        self.dropout = nn.Dropout(p_drop)
        self.norm_in = nn.LayerNorm(d_model)

    def extend_pe(self, n_pos_needed):
        if n_pos_needed <= self.pe.size(1): return
        with torch.no_grad():
            self.register_buffer("pe", sinusoidal_positions(n_pos_needed, self.pe.size(-1), device=self.pe.device), persistent=False)

    def forward(self, tok_ids, pos_ids, feat_rows, lengths,
                edge_index=None, edge_type=None):
        B, N = tok_ids.shape
        mask = torch.arange(N, device=tok_ids.device)[None, :] < lengths[:, None]  # [B,N] True for valid

        # embeddings
        x = torch.cat([
            self.emb_tok(tok_ids),
            self.emb_pos(pos_ids),
            self.W_feat(feat_rows)
        ], dim=-1)  # [B,N,d_model]
        x = self.norm_in(x)
        self.extend_pe(N)
        x = x + self.pe[:, :N, :]

        # sequence encoder
        h = self.enc_seq(x, src_key_padding_mask=~mask) if isinstance(self.enc_seq, nn.TransformerEncoder) else self.enc_seq(x)

        # optional simple GNN message passing over typed edges
        if self.use_gnn and edge_index is not None and edge_type is not None and edge_index.numel() > 0:
            # edge_index: [2,E] in global (batched) space; assume caller already offset indices
            src, dst = edge_index  # [E]
            rel = edge_type
            rel_e = self.edge_rel_emb(rel)                 # [E,d]
            m = torch.cat([h.view(-1, h.size(-1))[src], rel_e], dim=-1)  # [E, 2d]
            m = torch.tanh(self.msg_proj(m))               # [E,d]
            # aggregate messages to dst (simple scatter-add)
            H = h.view(-1, h.size(-1))
            agg = torch.zeros_like(H)
            agg.index_add_(0, dst, m)
            h = self.msg_norm(H + self.dropout(agg)).view(B, N, -1)

        # sentence-level roots via masked mean
        h_masked = h * mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
        h_sent = h_masked.sum(dim=1) / denom
        z = self.enc_lat(h_sent)  # [B,32]

        # reconstruct average interpretable features as an auxiliary target
        feat_hat = self.dec_feat(z)  # [B,K]

        # token roles
        role_logits = self.role_head(h)  # [B,N,C]
        return {"z": z, "feat_hat": feat_hat, "role_logits": role_logits, "mask": mask}

    # ---- losses you can call in your training loop ----
    def loss_reconstruction(self, feat_hat, feat_rows, mask=None, reduction="mean"):
        # compare sentence-mean(features) with feat_hat
        if mask is None:
            sent_target = feat_rows.mean(dim=1)
        else:
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
            sent_target = (feat_rows * mask.unsqueeze(-1)).sum(dim=1) / denom
        # If features are 0/1 probabilities, BCEWithLogits makes sense:
        return F.binary_cross_entropy_with_logits(feat_hat, sent_target, reduction=reduction)

    def loss_roles(self, role_logits, role_labels, mask):
        # role_labels: Long[B,N], mask: Bool[B,N]
        if self.use_crf:
            ll = self.crf(role_logits, role_labels, mask=mask, reduction='mean')
            return -ll
        else:
            C = role_logits.size(-1)
            loss = F.cross_entropy(role_logits.view(-1, C), role_labels.view(-1), reduction='none')
            loss = loss.view(role_labels.shape)
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp_min(1)

# ---------------- COMPLETE TRAINING SYSTEM ----------------
def train_step(model, batch, optimizer, w_rec=1.0, w_roles=1.0):
    model.train()
    out = model(batch["tok_ids"], batch["pos_ids"], batch["feat_rows"], batch["lengths"],
                batch.get("edge_index"), batch.get("edge_type"))
    loss = 0.0
    if w_rec:
        loss_rec = model.loss_reconstruction(out["feat_hat"], batch["feat_rows"], out["mask"])
        loss = loss + w_rec * loss_rec
    if w_roles and "role_labels" in batch:
        loss_roles = model.loss_roles(out["role_logits"], batch["role_labels"], out["mask"])
        loss = loss + w_roles * loss_roles
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}

class DataLoader:
    """Advanced data loader for World Engine with dynamic batching and feature engineering."""
    def __init__(self, dataset, batch_size=32, shuffle=True, vocab_size=10000, max_length=128):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.feature_extractor = FeatureExtractor()

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = self._create_batch([self.dataset[idx] for idx in batch_indices])
            yield batch

    def _create_batch(self, samples):
        batch_size = len(samples)
        max_len = min(max(len(s['tokens']) for s in samples), self.max_length)

        tok_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        pos_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        feat_rows = torch.zeros(batch_size, max_len, 50)  # 50 interpretable features
        lengths = torch.tensor([min(len(s['tokens']), max_len) for s in samples])

        for i, sample in enumerate(samples):
            length = lengths[i].item()
            tok_ids[i, :length] = torch.tensor(sample['tokens'][:length])
            pos_ids[i, :length] = torch.tensor(sample.get('pos_tags', list(range(length))))

            # Extract interpretable features
            features = self.feature_extractor.extract(sample['text'][:length])
            feat_rows[i, :length] = features

        return {
            "tok_ids": tok_ids,
            "pos_ids": pos_ids,
            "feat_rows": feat_rows,
            "lengths": lengths
        }

class FeatureExtractor:
    """Extracts 50 interpretable linguistic and semantic features."""
    def __init__(self):
        self.pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PREP', 'DET', 'CONJ', 'NUM', 'PRON', 'PART']
        self.semantic_fields = ['PERSON', 'PLACE', 'TIME', 'EVENT', 'OBJECT', 'ABSTRACT', 'EMOTION', 'ACTION']

    def extract(self, text_tokens):
        """Extract 50-dimensional feature vector for each token."""
        features = []
        for token in text_tokens:
            feat_vec = torch.zeros(50)

            # Basic linguistic features (0-19)
            feat_vec[0] = len(token) / 10.0  # normalized length
            feat_vec[1] = float(token.isupper())
            feat_vec[2] = float(token.islower())
            feat_vec[3] = float(token.istitle())
            feat_vec[4] = float(token.isdigit())
            feat_vec[5] = float(any(c.isdigit() for c in token))
            feat_vec[6] = float('-' in token)
            feat_vec[7] = float('_' in token)
            feat_vec[8] = sum(1 for c in token if c.isupper()) / max(len(token), 1)
            feat_vec[9] = sum(1 for c in token if c.islower()) / max(len(token), 1)

            # Morphological features (10-19)
            feat_vec[10] = float(token.endswith('ing'))
            feat_vec[11] = float(token.endswith('ed'))
            feat_vec[12] = float(token.endswith('er'))
            feat_vec[13] = float(token.endswith('est'))
            feat_vec[14] = float(token.endswith('ly'))
            feat_vec[15] = float(token.startswith('re'))
            feat_vec[16] = float(token.startswith('un'))
            feat_vec[17] = float(token.startswith('pre'))
            feat_vec[18] = float(token.endswith('tion'))
            feat_vec[19] = float(token.endswith('ness'))

            # Semantic features (20-39)
            for i, field in enumerate(self.semantic_fields):
                feat_vec[20 + i] = self._semantic_similarity(token, field)

            # Positional and contextual features (40-49)
            feat_vec[40] = float(token in ['the', 'a', 'an'])  # articles
            feat_vec[41] = float(token in ['and', 'or', 'but'])  # conjunctions
            feat_vec[42] = float(token in ['in', 'on', 'at', 'to', 'for'])  # prepositions
            feat_vec[43] = float(token in ['I', 'you', 'he', 'she', 'it', 'we', 'they'])  # pronouns
            feat_vec[44] = float(token.lower() in ['not', "n't", 'never', 'no'])  # negation
            feat_vec[45] = float(token.lower() in ['very', 'quite', 'really', 'extremely'])  # intensifiers
            feat_vec[46] = float(token.lower() in ['can', 'could', 'will', 'would', 'should', 'must'])  # modals
            feat_vec[47] = sum(1 for c in token if c in '.,!?;:') > 0  # punctuation
            feat_vec[48] = float(token.lower() in ['yes', 'no', 'maybe', 'perhaps'])  # response words
            feat_vec[49] = hash(token) % 100 / 100.0  # token hash feature

            features.append(feat_vec)

        return torch.stack(features) if features else torch.zeros(1, 50)

    def _semantic_similarity(self, token, semantic_field):
        """Simple semantic similarity based on common words."""
        field_words = {
            'PERSON': ['person', 'man', 'woman', 'child', 'people', 'human', 'individual'],
            'PLACE': ['place', 'city', 'country', 'home', 'location', 'area', 'region'],
            'TIME': ['time', 'day', 'year', 'month', 'hour', 'moment', 'period'],
            'EVENT': ['event', 'meeting', 'party', 'conference', 'celebration', 'ceremony'],
            'OBJECT': ['thing', 'object', 'item', 'tool', 'device', 'machine', 'equipment'],
            'ABSTRACT': ['idea', 'concept', 'thought', 'theory', 'principle', 'belief'],
            'EMOTION': ['happy', 'sad', 'angry', 'excited', 'calm', 'nervous', 'afraid'],
            'ACTION': ['run', 'walk', 'jump', 'work', 'play', 'create', 'build', 'move']
        }

        words = field_words.get(semantic_field, [])
        return float(token.lower() in words or any(w in token.lower() for w in words))

class WorldEngineTrainer:
    """Complete training system for World Engine with advanced features."""
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler=None, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0

    def train_epoch(self, w_rec=1.0, w_roles=1.0):
        """Train for one epoch with detailed metrics."""
        self.model.train()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}

            # Forward pass
            out = self.model(batch["tok_ids"], batch["pos_ids"], batch["feat_rows"], batch["lengths"])

            # Compute losses
            loss = 0.0
            if w_rec > 0:
                loss_rec = self.model.loss_reconstruction(out["feat_hat"], batch["feat_rows"], out["mask"])
                loss += w_rec * loss_rec

            if w_roles > 0 and "role_labels" in batch:
                loss_roles = self.model.loss_roles(out["role_logits"], batch["role_labels"], out["mask"])
                loss += w_roles * loss_roles

                # Compute accuracy
                preds = out["role_logits"].argmax(dim=-1)
                mask = out["mask"]
                correct = (preds == batch["role_labels"]) & mask
                correct_predictions += correct.sum().item()
                total_predictions += mask.sum().item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_samples += batch["tok_ids"].size(0)

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / max(total_predictions, 1)

        return avg_loss, accuracy

    def validate(self, w_rec=1.0, w_roles=1.0):
        """Validate the model with detailed metrics."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}

                out = self.model(batch["tok_ids"], batch["pos_ids"], batch["feat_rows"], batch["lengths"])

                loss = 0.0
                if w_rec > 0:
                    loss_rec = self.model.loss_reconstruction(out["feat_hat"], batch["feat_rows"], out["mask"])
                    loss += w_rec * loss_rec

                if w_roles > 0 and "role_labels" in batch:
                    loss_roles = self.model.loss_roles(out["role_logits"], batch["role_labels"], out["mask"])
                    loss += w_roles * loss_roles

                    preds = out["role_logits"].argmax(dim=-1)
                    mask = out["mask"]
                    correct = (preds == batch["role_labels"]) & mask
                    correct_predictions += correct.sum().item()
                    total_predictions += mask.sum().item()

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / max(total_predictions, 1)

        return avg_loss, accuracy

    def train(self, num_epochs, w_rec=1.0, w_roles=1.0, save_path=None):
        """Complete training loop with early stopping and checkpointing."""
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Training
            train_loss, train_acc = self.train_epoch(w_rec, w_roles)

            # Validation
            val_loss, val_acc = self.validate(w_rec, w_roles)

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping and checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                if save_path:
                    self.save_checkpoint(save_path, epoch, val_loss)
                    print(f"New best model saved to {save_path}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print("\nTraining completed!")
        return self.history

    def save_checkpoint(self, path, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['val_loss']

class WorldEngineEvaluator:
    """Comprehensive evaluation system for World Engine."""
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def evaluate_semantic_coherence(self, test_loader):
        """Evaluate semantic coherence of learned representations."""
        self.model.eval()
        coherence_scores = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                out = self.model(batch["tok_ids"], batch["pos_ids"], batch["feat_rows"], batch["lengths"])

                # Compute coherence based on reconstruction quality
                if "feat_hat" in out:
                    feat_hat = out["feat_hat"]
                    mask = out["mask"]

                    # Sentence-level feature targets
                    denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
                    sent_target = (batch["feat_rows"] * mask.unsqueeze(-1)).sum(dim=1) / denom

                    # Cosine similarity between predicted and target features
                    feat_hat_norm = F.normalize(feat_hat, p=2, dim=1)
                    sent_target_norm = F.normalize(sent_target, p=2, dim=1)
                    coherence = (feat_hat_norm * sent_target_norm).sum(dim=1)
                    coherence_scores.extend(coherence.cpu().tolist())

        return {
            'mean_coherence': sum(coherence_scores) / len(coherence_scores),
            'std_coherence': torch.tensor(coherence_scores).std().item(),
            'min_coherence': min(coherence_scores),
            'max_coherence': max(coherence_scores)
        }

    def analyze_latent_space(self, test_loader, num_samples=1000):
        """Analyze the learned latent space structure."""
        self.model.eval()
        latent_vectors = []

        with torch.no_grad():
            for batch in test_loader:
                if len(latent_vectors) >= num_samples:
                    break

                batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                out = self.model(batch["tok_ids"], batch["pos_ids"], batch["feat_rows"], batch["lengths"])

                if "z" in out:
                    latent_vectors.extend(out["z"].cpu().tolist())

        latent_vectors = latent_vectors[:num_samples]
        latent_tensor = torch.tensor(latent_vectors)

        # Compute statistics
        mean_vec = latent_tensor.mean(dim=0)
        std_vec = latent_tensor.std(dim=0)

        # Compute pairwise distances
        distances = torch.cdist(latent_tensor, latent_tensor)
        avg_distance = distances[distances > 0].mean()

        return {
            'dimensionality': latent_tensor.shape[1],
            'mean_vector': mean_vec.tolist(),
            'std_vector': std_vec.tolist(),
            'average_pairwise_distance': avg_distance.item(),
            'effective_dimensionality': (std_vec > 0.01).sum().item()
        }

def create_synthetic_dataset(num_samples=10000, vocab_size=5000, max_length=50):
    """Create a synthetic dataset for testing World Engine."""
    import random

    # Common word patterns
    prefixes = ['re', 'un', 'pre', 'dis', 'over', 'under', 'out', 'up']
    roots = ['work', 'play', 'build', 'create', 'move', 'think', 'learn', 'grow']
    suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness', 'ment']

    articles = ['the', 'a', 'an']
    prepositions = ['in', 'on', 'at', 'to', 'for', 'with', 'by']
    conjunctions = ['and', 'or', 'but', 'because', 'when', 'where', 'how']

    dataset = []

    for i in range(num_samples):
        # Generate sentence length
        length = random.randint(5, max_length)

        tokens = []
        text = []

        for j in range(length):
            if j == 0 or random.random() < 0.1:  # Start or random article
                if random.random() < 0.3:
                    word = random.choice(articles)
                else:
                    word = random.choice(roots)
            elif random.random() < 0.15:  # Conjunction
                word = random.choice(conjunctions)
            elif random.random() < 0.2:   # Preposition
                word = random.choice(prepositions)
            else:  # Regular word with possible morphology
                root = random.choice(roots)
                if random.random() < 0.3:  # Add prefix
                    word = random.choice(prefixes) + root
                else:
                    word = root
                if random.random() < 0.4:  # Add suffix
                    word = word + random.choice(suffixes)

            tokens.append(hash(word) % vocab_size)  # Convert to token ID
            text.append(word)

        # Generate synthetic role labels (IOB tagging)
        role_labels = []
        in_entity = False
        for k, word in enumerate(text):
            if word in articles + prepositions + conjunctions:
                role_labels.append(0)  # O - Outside
                in_entity = False
            elif not in_entity and random.random() < 0.3:
                role_labels.append(1)  # B - Beginning
                in_entity = True
            elif in_entity:
                if random.random() < 0.7:
                    role_labels.append(2)  # I - Inside
                else:
                    role_labels.append(0)  # O - Outside
                    in_entity = False
            else:
                role_labels.append(0)  # O - Outside

        dataset.append({
            'tokens': tokens,
            'text': text,
            'role_labels': role_labels,
            'pos_tags': [j % 10 for j in range(len(tokens))]  # Synthetic POS tags
        })

    return dataset

def main():
    """Main training and evaluation function."""
    print("Initializing World Engine Robust Training System...")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create synthetic dataset
    print("Creating synthetic dataset...")
    full_dataset = create_synthetic_dataset(num_samples=10000)

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))

    train_dataset = full_dataset[:train_size]
    val_dataset = full_dataset[train_size:train_size + val_size]
    test_dataset = full_dataset[train_size + val_size:]

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = WorldEngine(
        vocab_size=5000,
        d_model=256,
        k_feats=50,
        n_pos=100,
        n_rels=20,
        use_transformer=True,
        n_layers=4,
        n_heads=8,
        use_gnn=True,
        use_crf=True,
        num_role_labels=3
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Initialize trainer
    trainer = WorldEngineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    # Train model
    print("Starting training...")
    history = trainer.train(
        num_epochs=50,
        w_rec=1.0,
        w_roles=1.0,
        save_path="world_engine_best.pt"
    )

    # Evaluate model
    print("Evaluating model...")
    evaluator = WorldEngineEvaluator(model, device)

    coherence_metrics = evaluator.evaluate_semantic_coherence(test_loader)
    latent_analysis = evaluator.analyze_latent_space(test_loader)

    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Semantic Coherence: {coherence_metrics['mean_coherence']:.4f} ± {coherence_metrics['std_coherence']:.4f}")
    print(f"Latent Space Dimensionality: {latent_analysis['effective_dimensionality']}/{latent_analysis['dimensionality']}")
    print(f"Average Pairwise Distance: {latent_analysis['average_pairwise_distance']:.4f}")

    # Save final results
    results = {
        'history': history,
        'coherence_metrics': coherence_metrics,
        'latent_analysis': latent_analysis,
        'model_config': {
            'vocab_size': 5000,
            'd_model': 256,
            'k_feats': 50,
            'n_layers': 4,
            'n_heads': 8
        }
    }

    torch.save(results, 'training_results.pt')
    print("\nTraining completed successfully!")
    print("Results saved to 'training_results.pt'")
    print("Best model saved to 'world_engine_best.pt'")

if __name__ == "__main__":
    main()
