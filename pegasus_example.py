

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

import torch


src_text = input()


model_name = "google/pegasus-xsum"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = PegasusTokenizer.from_pretrained(model_name)


model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)


batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)


translated = model.generate(**batch)


tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

print(tgt_text)

