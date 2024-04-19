import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig
import os

base_path = 'llava-v1.6-vicuna-7b'
os.system(f'git clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b {base_path}')
os.system(f'cd {base_path} && git lfs pull')

# pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b') 非开发机运行此命令
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.8, session_len=8192)
pipe = pipeline('llava-v1.6-vicuna-7b', 
                backend_config=backend_config)

def model(image, text):
    if image is None:
        return [(text, "请上传一张图片。")]
    else:
        response = pipe((text, image)).text
        return [(text, response)]

demo = gr.Interface(fn=model, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs=gr.Chatbot())
demo.queue(1).launch()   
