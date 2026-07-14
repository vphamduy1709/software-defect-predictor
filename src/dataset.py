import torch
from torch.utils.data import Dataset

# =============================================================
# 5. VOCABULARY & 6. DATASET
# =============================================================
class Vocab:
    PAD_ID = 0; UNK_ID = 1

    def __init__(self, min_freq: int = 1):
        self.token2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2token = {0: "<PAD>", 1: "<UNK>"}
        self.min_freq = min_freq
        self._next    = 2
        self.freq     = {}

    def _collect(self, node: dict):
        t = node["token"]
        self.freq[t] = self.freq.get(t, 0) + 1
        for c in node["children"]:
            self._collect(c)

    def build(self, dataset: list) -> "Vocab":
        self.freq.clear()
        for item in dataset:
            self._collect(item["ast"])
        for tok, cnt in sorted(self.freq.items(), key=lambda x: -x[1]):
            if cnt >= self.min_freq and tok not in self.token2id:
                self.token2id[tok] = self._next
                self.id2token[self._next] = tok
                self._next += 1
        print(f"  Vocab: {len(self.token2id)} tokens (min_freq={self.min_freq})")
        return self

    def get(self, tok: str) -> int:
        return self.token2id.get(tok, self.UNK_ID)

    def __len__(self):
        return len(self.token2id)

def tree_to_tensors(node_dict: dict, vocab: Vocab):
    nodes, edges = [], []
    stack = [(node_dict, None)]
    while stack:
        node, parent = stack.pop()
        cur = len(nodes)
        nodes.append(vocab.get(node["token"]))
        if parent is not None:
            edges.append((parent, cur))
        for child in reversed(node["children"]):
            stack.append((child, cur))
    if not edges:
        edges = [(0, 0)]
    return (torch.tensor(nodes, dtype=torch.long),
            torch.tensor(edges, dtype=torch.long))

class TreeDataset(Dataset):
    def __init__(self, data: list, vocab: Vocab):
        self._cache = []
        for item in data:
            nodes, edges = tree_to_tensors(item["ast"], vocab)
            self._cache.append({
                "nodes": nodes,
                "edges": edges,
                "label": torch.tensor(item["bug_label"], dtype=torch.long)
            })

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        return self._cache[idx]