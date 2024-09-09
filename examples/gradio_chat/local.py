import llama_cpp
import llama_cpp.llama_tokenizer
import uuid
import gradio as gr
from time import sleep
from guard import guard
from guard import logging_utils

llama = llama_cpp.Llama.from_pretrained(
    repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="*Q4_K_M.gguf",
    n_ctx=1024,
    verbose=False,
)

model = "gpt-3.5-turbo"

convo_id = str(uuid.uuid4())[:8]
logger = logging_utils.build_logger("convo_log", f"convo_log_{convo_id}.log")


def predict(message, history):
    messages = []

    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

    logger.info(f"user message:{message}")
    if not guard.is_permitted(message):
        out_of_scope_message = "This prompt is out of scope for your role."
        response = out_of_scope_message.split()
        logger.info(f"is_permitted: False")
        text = ""

        for chunk in response:
            sleep(0.05)
            content = chunk
            if content:
                text += content + " "
                yield text
    else:
        messages.append({"role": "user", "content": message})

        response = llama.create_chat_completion_openai_v1(
            model=model, messages=messages, stream=True
        )
        logger.info(f"is_permitted: True")
        text = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                text += content
                yield text


js = """function () {
  gradioURL = window.location.href
  if (!gradioURL.endsWith('?__theme=dark')) {
    window.location.replace(gradioURL + '?__theme=dark');
  }
}"""

css = """
footer {
    visibility: hidden;
}
full-height {
    height: 100%;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), js=js, css=css, fill_height=True) as demo:
    gr.ChatInterface(
        predict,
        fill_height=True,
        examples=[
            "What is the capital of France?",
            "Who was the first person on the moon?",
        ],
    )


if __name__ == "__main__":
    demo.queue().launch()
