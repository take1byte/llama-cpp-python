import llama_cpp
import llama_cpp.llama_tokenizer

import gradio as gr

llama = llama_cpp.Llama.from_pretrained(
    repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="*Q4_K_M.gguf",
    n_ctx=1024,
    verbose=False,
)

model = "gpt-3.5-turbo"


def predict(message, history):
    messages = []

    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

    gr.Info(f"user input: {message}")

    messages.append({"role": "user", "content": message})

    response = llama.create_chat_completion_openai_v1(
        model=model, messages=messages, stream=True
    )

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
