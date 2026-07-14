import torch
import torch.nn as nn
from collections import deque


class TreeLSTMCell(nn.Module):
    def __init__(self, x_dim: int, h_dim: int):
        super().__init__()
        self.h_dim = h_dim
        self.W_i = nn.Linear(x_dim, h_dim);  self.U_i = nn.Linear(h_dim, h_dim, bias=False)
        self.W_f = nn.Linear(x_dim, h_dim);  self.U_f = nn.Linear(h_dim, h_dim, bias=False)
        self.W_o = nn.Linear(x_dim, h_dim);  self.U_o = nn.Linear(h_dim, h_dim, bias=False)
        self.W_u = nn.Linear(x_dim, h_dim);  self.U_u = nn.Linear(h_dim, h_dim, bias=False)

    def forward(self, x, child_h, child_c):
        dev = x.device
        if child_h.shape[0] == 0:
            child_h = torch.zeros(1, self.h_dim, device=dev)
            child_c = torch.zeros(1, self.h_dim, device=dev)

        h_sum = child_h.sum(0, keepdim=True)                     
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))         
        f = torch.sigmoid(self.W_f(x) + self.U_f(child_h))       
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))         
        u = torch.tanh(   self.W_u(x) + self.U_u(h_sum))         
        c = i * u + (f * child_c).sum(0, keepdim=True)           
        h = o * torch.tanh(c)                                      
        return h, c

class TreeLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.h_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop      = nn.Dropout(dropout)
        self.cell      = TreeLSTMCell(embed_dim, hidden_dim)

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

        self.predictor = nn.Linear(hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def _build_children_map(self, edges, n):
        cmap = {i: [] for i in range(n)}
        for e in edges.tolist():
            p, c = int(e[0]), int(e[1])
            if p != c:
                cmap[p].append(c)
        return cmap

    def _bfs_order(self, cmap, n):
        order, visited = [], set()
        queue = deque([0])
        while queue:
            idx = queue.popleft()
            if idx in visited:
                continue
            visited.add(idx)
            order.append(idx)
            for c in cmap[idx]:
                queue.append(c)
        return order

    def _bottom_up(self, nodes, edges):
        n   = nodes.shape[0]
        dev = nodes.device
        cmap  = self._build_children_map(edges, n)
        order = self._bfs_order(cmap, n)
        h_store, c_store, losses = {}, {}, []

        for idx in reversed(order):          
            x = self.drop(self.embedding(nodes[idx]).unsqueeze(0)) 
            children = [c for c in cmap[idx] if c in h_store]

            if not children:
                child_h = torch.zeros(0, self.h_dim, device=dev)
                child_c = torch.zeros(0, self.h_dim, device=dev)
            else:
                child_h = torch.cat([h_store[c] for c in children], 0)  
                child_c = torch.cat([c_store[c] for c in children], 0)

            h_ctx  = (child_h.sum(0, keepdim=True)
                      if child_h.shape[0] > 0
                      else torch.zeros(1, self.h_dim, device=dev))
            logits = self.predictor(h_ctx)                            
            losses.append(self.criterion(logits, nodes[idx].unsqueeze(0)))

            h, c = self.cell(x, child_h, child_c)
            h_store[idx], c_store[idx] = h, c

        return h_store, losses

    def forward(self, nodes, edges):
        h_store, losses = self._bottom_up(nodes, edges)

        all_h   = torch.cat(list(h_store.values()), 0)   
        scores  = self.attn(all_h)                          
        weights = torch.softmax(scores, 0)                  
        repr_   = (weights * all_h).sum(0, keepdim=True)   

        loss = (torch.stack(losses).mean()
                if losses else torch.tensor(0.0, device=nodes.device))

        return repr_, loss