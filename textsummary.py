import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline
try:

    model_path = "sshleifer/distilbart-cnn-12-6"
    text_summary = pipeline("summarization", model=model_path, dtype=torch.float16)

    # text = """Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2021; as of December 2025, Forbes estimates his net worth to be around US$754 billion.
    # Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue 
    # business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002."""
    
    # summary = text_summary(text, max_length=130, min_length=30, do_sample=False)
    # print("Summary: ", summary[0]['summary_text']) 

    def summarize(input_text):
        summary = text_summary(input_text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    gr.close_all()

    demo = gr.Interface(fn=summarize,
                        inputs=[gr.Textbox(label="Input Text to Summarize", lines=6)], 
                        outputs=[gr.Textbox(label="Summarized Text", lines=4)], 
                        title="Text Summarization with DistilBART",
                        description="THIS APPLICATION WILL BE USED TO SUMMARIZE THE TEXT.")
    demo.launch(share=True)

except Exception as e:

    print("Error loading model: ", e)
