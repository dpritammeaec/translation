#!/usr/bin/python

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from nltk.tokenize import sent_tokenize, word_tokenize
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'

# download and save model
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")

# import tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

while True:
    # input_text = ["Elon Musk sells $8.5 billion in Tesla stock", 
    #             "I'm a professional academic and research writer.",
    #             "Get a job in US and work in Germany"]
    input_text_from_user=input("\n\nPlease enter input text: \n")
    input_text=sent_tokenize(input_text_from_user)
    model_inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # translate from English to Hindi
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
    )

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    print(f"translation: {translation}")

