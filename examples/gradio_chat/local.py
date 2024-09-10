import llama_cpp
import llama_cpp.llama_tokenizer
import uuid
import gradio as gr
from time import sleep
from guard import guard
from guard import logging_utils
from guard.constants import INPUT_FORMAT_MSG, OUT_OF_SCOPE_MSG
from guard.prompt_handler import split_instructions_and_data
from guard.response_handler import response_generator

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

    instr, data = split_instructions_and_data(message)

    logger.info(f"user message:{message}")
    logger.info(f"user instruction:{instr}; user data:{data}")

    if instr is None:
        for text in response_generator(INPUT_FORMAT_MSG):
            yield text
    else:
        message = instr if data is None else f"{instr} {data}"

        for user_message, assistant_message in history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})

        if guard.is_permitted(instr):
            logger.info(f"is_permitted: True")

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
        else:
            logger.info(f"is_permitted: False")
            for text in response_generator(OUT_OF_SCOPE_MSG):
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
            "I: summarize the document D: [document text]",
            "I: write an article about cybersecurity",
        ],
    )


if __name__ == "__main__":
    demo.queue().launch()
