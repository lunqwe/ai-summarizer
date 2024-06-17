from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Summarizer:
    def summarize(input: str) -> str:
        # https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary
        tokenizer = AutoTokenizer.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")
        model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")
        
        encoded_input = tokenizer(input, return_tensors="pt")
        summary_outputs = model.generate(**encoded_input, max_new_tokens=150)
        decoded_summary = tokenizer.decode(summary_outputs[0], skip_special_tokens=True)
        
        return decoded_summary
    
    # could be other function (example: functions for using different summarization models)