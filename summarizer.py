from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Summarizer:
    @staticmethod
    def summarize(input: str, model_name: str = "pszemraj/long-t5-tglobal-base-16384-book-summary") -> str:
        """
        Summarizes input text using a transformer-based model.

        Args:
            input (str): The input text to be summarized.
            model_name (str, optional): The name or path of the pre-trained model to use.
                                        Defaults to "pszemraj/long-t5-tglobal-base-16384-book-summary".

        Returns:
            str: The generated summary of the input text.
        """
        # Load tokenizer and model from Hugging Face model hub (https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        encoded_input = tokenizer(input, return_tensors="pt", max_length=1024, truncation=True)
        summary_outputs = model.generate(**encoded_input, max_new_tokens=150)
        
        # Decode the generated summary text and remove special tokens
        decoded_summary = tokenizer.decode(summary_outputs[0], skip_special_tokens=True)
        
        return decoded_summary