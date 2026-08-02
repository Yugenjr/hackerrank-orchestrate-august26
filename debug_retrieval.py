import pandas as pd
import numpy as np
from data_loader import DataLoader
from feature_engineering import ContextBuilder
from retrieval import RetrievalEngine, QueryExpander

loader = DataLoader('dataset')
loader.load_all()
ctx_builder = ContextBuilder(loader)
ret_engine = RetrievalEngine(loader, use_mocks=False)

df = pd.read_csv('dataset/sample_messages.csv')
row = df.iloc[0].to_dict()
payload = {str(k): v for k, v in row.items() if pd.notnull(v)}

features = ctx_builder.build_context(payload)
query_norm = QueryExpander.normalize(ret_engine._get_text_payload(payload))

# Let's just find message_0001
history = loader.data_dicts['message_history.csv']['message_0001']
hist_norm = QueryExpander.normalize(ret_engine._get_text_payload(history))

print("Query:", query_norm)
print("Hist:", hist_norm)

score = ret_engine.cross_encoder.predict([[query_norm, hist_norm]])
print("Raw score:", score)
print("Sigmoid:", 1 / (1 + np.exp(-score)))
