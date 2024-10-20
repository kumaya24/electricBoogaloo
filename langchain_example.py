# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("1231czx/llama3_it_ultra_list_and_bold500")
model = AutoModelForSequenceClassification.from_pretrained("1231czx/llama3_it_ultra_list_and_bold500")