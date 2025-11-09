def transformer_make_prediction(generator,tokenizer, text):
    result = generator(text, max_new_tokens=20, do_sample=True, top_k=50, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    print('TNSF result: '+result[0]["generated_text"].replace("\n", " ").strip())
    pass