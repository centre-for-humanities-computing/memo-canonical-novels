class AttentionAggregator(nn.Module):
    def __init__(self, input_dim):
        super(AttentionAggregator, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, paragraph_embeddings):
        # Compute attention scores
        attention_scores = self.attention(paragraph_embeddings).squeeze(-1)  # shape: (num_paragraphs,)
        attention_weights = torch.softmax(attention_scores, dim=0)  # shape: (num_paragraphs,)
        
        # Weighted sum of paragraph embeddings
        weighted_sum = torch.sum(paragraph_embeddings * attention_weights.unsqueeze(-1), dim=0)
        return weighted_sum

# Instantiate the aggregator and get the final representation
aggregator = AttentionAggregator(input_dim=768)  # XLM-RoBERTa base model has 768 dimensions
novel_representation = aggregator(paragraph_embeddings)  # shape: (embedding_dim,)
